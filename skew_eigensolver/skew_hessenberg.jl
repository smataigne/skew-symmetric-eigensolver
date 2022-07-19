using LinearAlgebra
using LoopVectorization

include("round.jl")
include("householder.jl")


@views function skewhess!(A)
    n=size(A,1)
    atmp = similar(A,n)
    vtmp = similar(atmp)
    for i=1:n-2
        v,tau = householder_reflector!(A[i+1:end,i], vtmp[i+1:end],n-i)

        A[i+1,i] -= tau*v[1]*(transpose(v)*A[i+1:end,i])
        for j=i+2:n
            A[j,i]=v[j-i]
        end
        A[i,i+1]=-A[i+1,i]
        for j=i+2:n
            A[i,j]=0#Purely esthetic
        end

        leftHouseholder!(A[i+1:end,i+1:end],v,atmp[i+1:end],tau)

        s = mul!(atmp[i+1:end], A[i+1:end,i+1:end], v)
        A[i+1,i+1]=0
        for j=i+2:n
            @inbounds A[j,j]=0
            @inbounds for k=i+1:j-1
                A[j,k] -= tau*s[j-i]*v[k-i]
                A[k,j]  = -A[j,k]
            end
        end
    end
    return
end

    
