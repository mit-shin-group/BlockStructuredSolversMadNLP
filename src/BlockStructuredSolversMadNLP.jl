module BlockStructuredSolversMadNLP

using MadNLP

import CUDA: CuVector, @cuda, blockIdx, blockDim, threadIdx, cld
import CUDA.CUSPARSE: CuSparseMatrixCSC
import BlockStructuredSolvers: BlockTriDiagData, initialize, factorize!, solve!

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

    split_block_tridiag(solver.inner.A_vec, solver.inner.B_vec, solver.tril, solver.inner.n)
    
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

# ────────────────────────────────────────────────────────────────────────────────────
# helper functions to extract block tridiagonal structure
# ────────────────────────────────────────────────────────────────────────────────────

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

# ─────────────────────────  kernel: colPtr → colIdx  ──────────────────────
function k_expand!(colIdx, colPtr, ncol::Int32)
    j0 = (blockIdx().x - 1) * blockDim().x + threadIdx().x - 1   # 0-based col
    if j0 < ncol
        first = colPtr[Int32(j0 + 1)]
        last  = colPtr[Int32(j0 + 2)] - 1
        pos   = first
        @inbounds while pos ≤ last
            colIdx[pos] = j0 + 1        # store 1-based column number
            pos += 1
        end
    end
    return
end

_expand(ptr, nnz, ncol) = begin
    idx = CuVector{eltype(ptr)}(undef, nnz)
    @cuda threads=128 blocks=cld(ncol,128) k_expand!(idx, ptr, Int32(ncol))
    idx                            # 1-based column numbers
end

# ─────────────────────────  main extractor  ───────────────────────────────
"""
    D, B = split_block_tridiag(A_gpu, b)

* `A_gpu` — cuSPARSE CSC storing **only the lower triangle** of a symmetric
  block-tridiagonal matrix.
* `b`     — block size.

Returns  

* `D[b,b,nb]`   — full diagonal blocks (last padded with identity if needed)  
* `B[b,b,nb-1]` — **upper** off-diagonal blocks (already transposed)
"""
function split_block_tridiag(vecD, vecB, A::CuSparseMatrixCSC{Tv,Ti}, b::Integer) where {Tv,Ti<:Integer}
    n    = size(A,1)
    nb   = cld(n, b)                       # ceil division
    nnz  = length(A.nzVal)
    ncol = size(A,2)
    bTi  = Ti(b);   b2Ti = bTi*bTi

    # 1. raw buffers on GPU (1-based)
    row1   = A.rowVal    
    colPtr = A.colPtr
    col1   = _expand(colPtr, nnz, ncol)
    nzval  = A.nzVal                       # already CuArray

    # 2. 0-based temporaries
    row0 = row1 .- one(Ti)
    col0 = col1 .- one(Ti)

    br  = row0 .÷ bTi;   lr = row0 .% bTi   # block row / local row
    bc  = col0 .÷ bTi;   lc = col0 .% bTi   # block col / local col
    lin = lr .+ lc .* bTi                   # 0-based index in b×b

    diag = br .== bc
    low  = br .== bc .+ one(Ti)

    # -------- indices & values for D (duplicate upper part) -----------------
    idx_lo = lin[diag] .+ br[diag] .* b2Ti .+ 1
    idx_up = lc[diag] .+ lr[diag] .* bTi .+ br[diag] .* b2Ti .+ 1
    valD   = nzval[diag]

    unique = idx_lo .!= idx_up             # off-diagonal in block
    idxD   = vcat(idx_lo, idx_up[unique])
    valD   = vcat(valD,  valD[unique])

    # -------- indices & values for B (already upper) ------------------------
    idxB = lc[low] .+ lr[low] .* bTi .+ bc[low] .* b2Ti .+ 1
    valB = nzval[low]

    # -------- allocate & scatter (GPU broadcasts) ---------------------------
    vecD[idxD] .= valD
    vecB[idxB] .= valB

    # -------- pad last diagonal block if matrix size ragged -----------------
    s = n - (nb-1)*b                       # physical size of last block
    if s < b
        base   = (nb-1)*b*b + 1            # flat offset of that block
        diagID = base .+ ((s):b .- 1) .* (b+1)
        vecD[diagID] .= one(Tv)            # identity on padded rows/cols
    end

end


end