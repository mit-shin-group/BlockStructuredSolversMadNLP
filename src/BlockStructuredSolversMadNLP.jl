module BlockStructuredSolversMadNLP

using MadNLP

import CUDA: CuVector, @cuda, blockIdx, blockDim, threadIdx, cld, CUDABackend
import CUDA.CUSPARSE: CuSparseMatrixCSC
import BlockStructuredSolvers: BlockTriDiagData, initialize, factorize!, solve!
import KernelAbstractions: @kernel, @index, @Const, synchronize

export split_block_tridiag, TBDSolver, TBDSolverOptions

@kwdef mutable struct TBDSolverOptions <: MadNLP.AbstractOptions
    #TODO add ordering
end

mutable struct TBDSolver{T} <: MadNLP.AbstractLinearSolver{T}
    inner::Union{Nothing, BlockTriDiagData}
    tril::CuSparseMatrixCSC{T}

    opt::MadNLP.AbstractOptions
    logger::MadNLP.MadNLPLogger

    dstD::CuVector
    srcD::CuVector
    dstB::CuVector
    srcB::CuVector
    padIdx::CuVector
    lenD::Int
    lenB::Int
end

function TBDSolver(
    csc::CuSparseMatrixCSC{T};
    opt=TBDSolverOptions(),
    logger=MadNLP.MadNLPLogger(),
    ) where T

    #TODO ordering
    N, n = detect_spaces_and_divide_csc(csc)
    println("N, n = ", N, ", ", n)
    solver = initialize(N, n, eltype(csc), true)
    dstD, srcD, dstB, srcB, padIdx, lenD, lenB = make_maps(csc, n)

    return TBDSolver(solver, csc, opt, logger, dstD, srcD, dstB, srcB, padIdx, lenD, lenB)
end

function MadNLP.factorize!(solver::TBDSolver{T}) where T

    copyto!(solver.inner.A_vec, solver.inner.A_fill)
    memcopy!(CUDABackend())(solver.inner.A_vec, solver.tril.nzVal, solver.dstD, solver.srcD; ndrange=solver.lenD)
    memcopy!(CUDABackend())(solver.inner.B_vec, solver.tril.nzVal, solver.dstB, solver.srcB; ndrange=solver.lenB)
    synchronize(CUDABackend())

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

# ------------------------------------------------------------------
# kernel: colPtr → colIdx   (values 1-based)
# ------------------------------------------------------------------
function k_expand!(idx, ptr, ncol::Int32)
    j0 = (blockIdx().x-1)*blockDim().x + threadIdx().x - 1        # 0-based col
    if j0 < ncol
        f = ptr[j0+1];  l = ptr[j0+2] - 1
        while f ≤ l
            @inbounds idx[f] = j0 + 1                             # 1-based col
            f += 1
        end
    end
end

_expand(ptr, nnz, ncol) = begin
    idx = CuVector{eltype(ptr)}(undef, nnz)
    @cuda threads=128 blocks=cld(ncol,128) k_expand!(idx, ptr, Int32(ncol))
    idx
end

# ------------------------------------------------------------------
# build GPU index vectors
# ------------------------------------------------------------------
function make_maps(A::CuSparseMatrixCSC{Tv,Ti}, b::Integer) where {Tv,Ti<:Integer}
    n      = size(A,1)
    nb     = cld(n, b)                         # ceil(n/b)  – ragged last blk OK
    nnz    = length(A.nzVal)
    bTi    = Ti(b);   b2Ti = bTi*bTi
    row1   = A.rowVal
    col1   = _expand(A.colPtr, nnz, size(A,2))
    nzids  = CuVector(Ti.(1:nnz))              # 1…nnz on GPU

    # 0-based temporaries
    row0 = row1 .- one(Ti)
    col0 = col1 .- one(Ti)
    br   = row0 .÷ bTi;  lr = row0 .% bTi
    bc   = col0 .÷ bTi;  lc = col0 .% bTi
    lin  = lr .+ lc .* bTi

    diag = br .== bc
    low  = br .== bc .+ one(Ti)

    # --- D (full diagonal blocks) ------------------------------------------
    dst_lo = lin[diag] .+ br[diag] .* b2Ti .+ 1
    dst_up = lc[diag] .+ lr[diag] .* bTi .+ br[diag] .* b2Ti .+ 1
    src_lo = nzids[diag]

    keep   = dst_lo .!= dst_up                     # off-diagonal in block
    dstD   = vcat(dst_lo, dst_up[keep])
    srcD   = vcat(src_lo, src_lo[keep])            # same value for upper copy

    # --- B (upper off-diagonal) --------------------------------------------
    dstB = lc[low] .+ lr[low] .* bTi .+ bc[low] .* b2Ti .+ 1
    srcB = nzids[low]

    # --- padding indices for identity in last block ------------------------
    s = n - (nb-1)*b                              # physical size of last block
    if s == b
        padIdx = CuVector{Ti}(undef, 0)           # nothing to pad
    else
        kvec   = CuVector(Ti.(s:b-1))             # missing local rows/cols
        padIdx = (kvec .+ kvec .* bTi) .+ Ti(nb-1)*b2Ti .+ 1
    end

    return dstD, srcD, dstB, srcB, padIdx, length(dstD), length(dstB)
end

@kernel function memcopy!(out, @Const(nzval), @Const(dst), @Const(src))
    k = @index(Global)
    @inbounds out[dst[k]] = nzval[src[k]]
end


end