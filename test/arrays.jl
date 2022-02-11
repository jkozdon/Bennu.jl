@testset "arrays" begin
    @test arraytype(ones(3, 5)) === Array
    if CUDA.has_cuda_gpu()
        @test arraytype(CuArray(ones(3, 5))) === CuArray
    end

    @test [1, 3, 1, 2, 1, 2] == Bennu.numbercontiguous([10, 29, 10, 23, 10, 23])
    TAs = ((Float64,  Array), (BigFloat, Array))
    if CUDA.has_cuda_gpu()
        TAs = (TAs..., (Float32, CuArray))
    end
    for (T, A) in TAs
        a = BennuArray(undef, (a = T,), A, ())
        @test size(a) == ()

        a = BennuArray(undef, (a = T, ), A, (3,))
        @test size(a) == (3,)
        @test eltype(a) == NamedTuple{(:a, ), Tuple{T}}
        @test Tullio.storage_type(a) <: A

        a = BennuArray(undef, (a = T,), A, (3,4))
        @test size(a) == (3,4)
        @test eltype(a) == NamedTuple{(:a, ), Tuple{T}}
        @test Tullio.storage_type(a) <: A

        a = BennuArray(undef, (a=SVector{2, T}, b=T), A, (3,4))
        @test size(a) == (3,4)
        @test eltype(a) == NamedTuple{(:a, :b), Tuple{SVector{2, T}, T}}
        @test Tullio.storage_type(a) <: A

        a = BennuArray(undef, SVector{2, T}, A, (3,4))
        @test size(a) == (3,4)
        @test eltype(a) == SVector{2, T}
        @test Tullio.storage_type(a) <: A

    end
end
