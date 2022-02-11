struct BennuArray{T, N, C, I, AT} <: AbstractArray{T, N}
    structarray::StructArray{T, N, C, I}
    data::AT
    function BennuArray(structarray::StructArray{T, N, C, I}) where {T, N, C, I}
        # Get the parent data array
        data = structarray
        while data isa StructArray
            data = parent(components(data)[1])
        end
        AT = typeof(data)

        # Check that all data in the structarray are backed by same parent array
        @assert _validate_fieldarray(structarray, data)
        return new{T, N, C, I, AT}(structarray, data)
    end
end
Base.size(f::BennuArray) = size(f.structarray)
Base.@propagate_inbounds function Base.getindex(f::BennuArray, a...)
    return getindex(f.structarray, a...)
end
Base.@propagate_inbounds function Base.setindex!(f::BennuArray, a...)
    setindex!(f.structarray, a...)
end
Tullio.storage_type(a::BennuArray) = Tullio.storage_type(a.structarray)
components(a::BennuArray) = components(a.structarray)
_validate_fieldarray(a::StructArray, data) =
all(_validate_fieldarray.(values(components(a)), Ref(data)))
_validate_fieldarray(a, data) = pointer(parent(a)) == pointer(data)

arraytype(A) = Tullio.storage_type(A) <: CuArray ? CuArray : Array
arraytype(::Type{T}) where {T} = Array
arraytype(::Type{<:CuArray}) = CuArray
arraytype(::Type{<:CUDA.Adaptor}) = CuArray

components(a::AbstractArray) = (a,)

function numbercontiguous(A; by=identity)
    p = sortperm(A; by=by)
    notequalprevious = fill!(similar(p, Bool), false)
    @tullio notequalprevious[i] =
        @inbounds(begin by(A[p[i]]) != by(A[p[i-1]]) end) (i in 2:length(p))

    B = similar(p)
    B[p] .= cumsum(notequalprevious) .+ 1

    return B
end

_numfields(::Type) = 1
_numfields(::Type{S}) where {S <: SArray} = length(S)
_numfields(T::Tuple) = sum(map(_numfields, T))
_numfields(N::NamedTuple) = sum(map(_numfields, N))

_fieldtype(::Type{S}) where {S} = eltype(S)
_fieldtype(T::Tuple) = promote_type(map(_fieldtype, T)...)
_fieldtype(N::NamedTuple) = promote_type(map(_fieldtype, N)...)

function BennuArray(::UndefInitializer, S, ::Type{A}, dims::Dims) where {A}
    N = _numfields(S)
    T = _fieldtype(S)

    S isa Type && S <: Number && error("BennuArray requires `!(S <: Number)`")

    if length(dims) == 0
        d = (N, )
    elseif length(dims) == 1
        d = (dims[1], N)
    else
        d = (dims[1:end-1]..., N, dims[end])
    end

    data = A{T}(undef, d)
    dataviews = ntuple(N) do i
        offset = length(dims) > 1 ? 0 : 1
        viewtuple = ntuple(j->(j==length(dims)+offset ? i : :), length(d))
        return view(data, viewtuple...)
    end

    structarrayn = _fieldarray(S, dataviews)
    BennuArray(structarrayn)
end

function _ckfieldargs(S, data::Tuple)
    if _numfields(S) != length(data)
        throw(ArgumentError("Number of data fields is incorrect."))
    end
    T = _fieldtype(S)
    dims = size(first(data))
    for d in data
        if eltype(d) != T
            throw(ArgumentError("Data arrays do not have the same eltype."))
        end
        if size(d) != dims
            throw(ArgumentError("Data arrays do not have the same size."))
        end
    end
end

function _fieldarray(::Type{S}, data::Tuple) where {S}
    d = only(data)
    if S != eltype(d)
        throw(ArgumentError("Data array does not have the correct eltype."))
    end

    return d
end

function _fieldarray(::Type{S}, data::Tuple) where {S <: SArray}
    _ckfieldargs(S, data)
    return StructArray{S}(data)
end

function _fieldarray(S::NamedTuple, data::Tuple)
    _ckfieldargs(S, data)

    offsets = cumsum((1, map(_numfields, S)...))
    fields = ntuple(length(S)) do i
        _fieldarray(S[i], data[offsets[i]:offsets[i+1]-1])
    end

    return StructArray(NamedTuple{keys(S)}(fields))
end
