struct BennuArray{T, N, C, I, AT} <: AbstractArray{T, N}
    structarray::StructArray{T, N, C, I}
    data::AT
    function BennuArray(structarray::StructArray{T, N, C, I}) where {T, N, C, I}
        # Get the parent data array
        data = structarray
        while data isa Union{StructArray, Base.ReshapedArray, SubArray}
            data = if data isa StructArray
                parent(components(data)[1])
            else
                parent(data)
            end
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

_validate_fieldarray(a::Union{Base.ReshapedArray, SubArray}, data) =
    _validate_fieldarray(parent(a), data)

_validate_fieldarray(a, data) = pointer(a) == pointer(data)

Base.similar(s::BennuArray, sz::Base.DimOrInd...) = similar(s, Base.to_shape(sz))
Base.similar(s::BennuArray) = similar(s, Base.to_shape(axes(s)))
function Base.similar(s::BennuArray{T}, sz::Tuple) where {T}
    BennuArray(undef, T, arraytype(s), sz)
end
Base.reshape(s::BennuArray, d::Dims) = BennuArray(reshape(s.structarray, d))

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
_numfields(::Type{S}) where S <: NamedTuple = sum(map(_numfields, S.types))
_numfields(T::Tuple) = sum(map(_numfields, T))
_numfields(N::NamedTuple) = sum(map(_numfields, N))

_fieldtype(::Type{S}) where {S} = eltype(S)
_fieldtype(::Type{S}) where S <: BennuArray{T} where T = _fieldtype(T)
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

    structarray = _fieldarray(S, dataviews)
    BennuArray(structarray)
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

function _fieldarray(::Type{S} , data::Tuple) where {S <: NamedTuple}
    _ckfieldargs(S, data)

    offsets = cumsum((1, map(_numfields, S.types)...))
    fields = ntuple(length(S.types)) do i
        _fieldarray(S.types[i], data[offsets[i]:offsets[i+1]-1])
    end

    return StructArray(NamedTuple{S.parameters[1]}(fields))
end

# BennuArray Broadcast
# Based on StructArrays:
# https://github.com/JuliaArrays/StructArrays.jl/blob/8e67e4e778e3d8d188370f1203be5317d7bd6be7/src/structarray.jl
import Base.Broadcast: BroadcastStyle, ArrayStyle, AbstractArrayStyle, Broadcasted, DefaultArrayStyle

struct BennuArrayStyle{Style} <: AbstractArrayStyle{Any} end

@inline combine_style_types(::Type{A}, args...) where A<:AbstractArray =
    combine_style_types(BroadcastStyle(A), args...)
@inline combine_style_types(s::BroadcastStyle, ::Type{A}, args...) where A<:AbstractArray =
    combine_style_types(Broadcast.result_style(s, BroadcastStyle(A)), args...)
combine_style_types(s::BroadcastStyle) = s

array_types(::Type{BennuArray{T, N, C, I, AT}}) where {T, N, C, I, AT} = array_types(C)
array_types(::Type{NamedTuple{names, types}}) where {names, types} = types
array_types(::Type{TT}) where {TT<:Tuple} = TT

Base.@pure cst(::Type{BA}) where BA = combine_style_types(array_types(BA).parameters...)

BroadcastStyle(::Type{BA}) where BA<:BennuArray = BennuArrayStyle{typeof(cst(BA))}()

function Base.similar(bc::Broadcast.Broadcasted{BennuArrayStyle{S}},
                      ::Type{T}) where {S<:CUDA.CuArrayStyle,T}
    if isstructtype(T)
        BennuArray(undef, T, CuArray, Base.to_shape(axes(bc)))
    else
        return similar(CuArray{T}, axes(bc))
    end
end

function Base.similar(bc::Broadcast.Broadcasted{BennuArrayStyle{S}},
                      ::Type{T}) where {S, T}
    if isstructtype(T)
        BennuArray(undef, T, Array, Base.to_shape(axes(bc)))
    else
        return similar(Array{T}, axes(bc))
    end
end

# for aliasing analysis during broadcast
Base.dataids(u::BennuArray) = mapreduce(Base.dataids, (a, b) -> (a..., b...), values(components(u)), init=())
