module BlockStructuredSolversMadNLP

using MadNLP

import CUDA: CuMatrix, CuArray, CuVector, @cuda
import CUDA.CUSPARSE: CuSparseMatrixCSC
import BlockStructuredSolvers: BlockTriDiagData, factorize!, solve!

export split_block_tridiag, TBDSolver, TBDSolverOptions

@kwdef mutable struct TBDSolverOptions <: MadNLP.AbstractOptions
    #TODO add ordering
end

mutable struct TBDSolver{T} <: MadNLP.AbstractLinearSolver{T}
    inner::Union{Nothing, BlockTriDiagData}
    tril::CuSparseMatrixCSC{T}

    opt::MadNLP.AbstractOptions
    logger::MadNLP.MadNLPLogger
end

function TBDSolver(
    csc::CuSparseMatrixCSC{T};
    opt=TBDSolverOptions(),
    logger=MadNLP.MadNLPLogger(),
    ) where T

    #TODO ordering
    N, n = detect_spaces_and_divide_csc(csc)
    println("N, n = ", N, ", ", n)
    # N = 50
    # n = 2
    solver = initialize(N, n, eltype(csc), true)

    return TBDSolver(solver, csc, opt, logger)
end

function MadNLP.factorize!(solver::TBDSolver{T}) where T

    D, B = split_block_tridiag(solver.tril, solver.inner.n)
    for i = 1:size(D, 3)-1
        solver.inner.A_list[i] = D[:,:,i]
        solver.inner.B_list[i] = B[:,:,i]
    end
    solver.inner.A_list[end] = D[:,:,end]
    
    factorize!(solver.inner)
    return solver
end

function MadNLP.solve!(solver::TBDSolver{T}, d) where T

    copyto!(solver.inner.d, d)
    solve!(solver.inner, solver.inner.d_list)
    copyto!(d, view(solver.inner.d, 1:length(d)))

    return d
end

MadNLP.input_type(::Type{TBDSolver}) = :csc
MadNLP.default_options(::Type{TBDSolver}) = TBDSolverOptions()
MadNLP.improve!(M::TBDSolver) = false
MadNLP.is_supported(::Type{TBDSolver},::Type{Float32}) = true
MadNLP.is_supported(::Type{TBDSolver},::Type{Float64}) = true

#TODO intertia
MadNLP.is_inertia(M::TBDSolver) = true
function MadNLP.inertia(M::TBDSolver)
    n = size(M.tril, 1)
    return (n, 0, 0)
end

#TODO improve
MadNLP.improve!(M::TBDSolver) = false

#TODO introduce
MadNLP.introduce(M::TBDSolver) = "TBDSolver"

function detect_spaces_and_divide_csc(csc_matrix::CuSparseMatrixCSC{T}) where T
    # Get matrix dimensions
    num_rows, num_cols = size(csc_matrix, 1), size(csc_matrix, 2)
    
    # Copy GPU arrays to CPU to avoid scalar indexing
    colPtr_cpu = Array(csc_matrix.colPtr)
    rowVal_cpu = Array(csc_matrix.rowVal)
    
    # We only need to track the first column seen for each row to calculate span
    # Using typemax(Int) as a sentinel value for uninitialized entries
    first_col_seen = fill(typemax(Int), num_rows)
    
    # Track maximum span across all rows
    max_span = 0
    
    @inbounds for col in 1:num_cols
        for ptr in colPtr_cpu[col]:(colPtr_cpu[col+1]-1)
            row = rowVal_cpu[ptr]
            
            # If this is the first time seeing this row, record the column
            if first_col_seen[row] == typemax(Int)
                first_col_seen[row] = col
            end
            
            # Calculate current span and update max_span if needed
            # Current span = current column - first column + 1
            current_span = col - first_col_seen[row] + 1
            max_span = max(max_span, current_span)
        end
    end
    
    # Ensure we had at least some non-zero entries
    if max_span == 0
        return num_rows, 1
    end
    
    # Estimate block size
    result = ceil(Int, max_span * 2 / 3)
    
    # Return number of blocks and block size (ensuring result is at least 1)
    return ceil(Int, num_rows / max(1, result)), max(1, result)
end

function k_expand!(colIdx, colPtr, ncol)
    j0 = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1   # 0-based col
    if j0 < ncol
        first = colPtr[j0 + 1]
        last  = colPtr[j0 + 2] - 1
        @inbounds for k = first:last
            colIdx[k] = j0 + 1        # store 1-based column number
        end
    end
end

expand_colPtr(ptr, nnz, ncol) = begin
    colIdx = CuVector{eltype(ptr)}(undef, nnz)
    @cuda threads=128 blocks=cld(ncol,128) k_expand!(colIdx, ptr, ncol)
    colIdx
end

# ────────────────────────────
# main extractor
# ────────────────────────────
"""
    D, B = split_block_tridiag(A_gpu, b)

Extract

* `D[b,b,nb]` – diagonal blocks  
* `B[b,b,nb-1]` – **upper** off-diagonal blocks

from a `CuSparseMatrixCSC` that stores *only the lower triangle*
of a symmetric block-tridiagonal matrix with block size `b`.
"""
function split_block_tridiag(A::CuSparseMatrixCSC{Tv,Ti}, b::Integer) where {Tv,Ti<:Integer}
    n   = size(A,1);    nb  = div(n, b)                # assume n ≡ 0 (mod b)
    nnz = length(A.nzVal);   ncol = size(A,2)
    bTi, b2Ti = Ti(b), Ti(b*b)

    # 1. GPU buffers (1-based)
    row1   = A.rowVal
    colPtr = A.colPtr
    col1   = expand_colPtr(colPtr, nnz, ncol)
    nzval  = A.nzVal

    # 2. 0-based temporaries
    row0 = row1 .- one(Ti)
    col0 = col1 .- one(Ti)

    br  = row0 .÷ bTi;   lr = row0 .% bTi          # block row / local row
    bc  = col0 .÷ bTi;   lc = col0 .% bTi          # block col / local col
    lin = lr .+ lc .* bTi                          # 0-based linear index

    maskD = br .== bc                              # entries in diagonal blocks
    maskL = br .== bc .+ one(Ti)                   # strictly lower blocks

    # ---- indices for diagonal blocks: original + transpose -----------------
    idx_lo = lin[maskD] .+ br[maskD] .* b2Ti .+ 1                      # lower
    idx_up = lc[maskD] .+ lr[maskD] .* bTi .+ br[maskD] .* b2Ti .+ 1   # upper
    valD   = nzval[maskD]

    # upper & lower coincide on diagonal, keep unique
    diff   = idx_lo .!= idx_up
    idxD   = vcat(idx_lo, idx_up[diff])           # 1-based destination slots
    valD   = vcat(valD,  valD[diff])              # duplicate values as needed

    # ---- indices for off-diagonal blocks (already upper form) -------------
    idxB = lc[maskL] .+ lr[maskL] .* bTi .+ bc[maskL] .* b2Ti .+ 1
    valB = nzval[maskL]

    # ---- scatter on the GPU (broadcast assignment = race-free) ------------
    vecD = CUDA.zeros(Tv, b*b*nb)
    vecB = CUDA.zeros(Tv, b*b*(nb-1))
    vecD[idxD] .= valD
    vecB[idxB] .= valB

    return reshape(vecD, b,b,nb), reshape(vecB, b,b,nb-1)
end

end