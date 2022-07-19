using LinearAlgebra
using BenchmarkTools

include("householder.jl")
include("skew_hessenberg.jl")

@views function skewmv(A,v,y,n)
    y[1]=0
    @inbounds for j=2:n
        y[1]-=A[j,1]*v[j]
        #y[1]=muladd(A[j,1],-v[j],y[1])
    end
    @inbounds for i=2:n-1
        y[i]=0
        for j=1:i-1
            y[i]+=A[i,j]*v[j]
            #y[i]=muladd(A[i,j],v[j],y[i])
        end
        for j=i+1:n
            y[i]-=A[j,i]*v[j]
            #y[i]=muladd(A[i,j],-v[j],y[i])
        end
    end
    y[n]=0
    @inbounds for j=1:n-1
        y[n]+=A[n,j]*v[j]
        #y[n]=muladd(A[n,j],v[j],y[n])
    end

end


@views function latrd!(A,E,W,V,tau,n,nb)

    @inbounds for i=1:nb
        #update A[i:n,i]
    
        if i>1
            #A[i:n,i]+=A[i:n,1:i-1]*W[i,1:i-1]
            LinearAlgebra.BLAS.gemv!('n',1.0,A[i:n,1:i-1],W[i,1:i-1],1.0,A[i:n,i])
            #A[i:n,i]-=W[i:n,1:i-1]*A[i,1:i-1]
            LinearAlgebra.BLAS.gemv!('n',-1.0,W[i:n,1:i-1],A[i,1:i-1],1.0,A[i:n,i])
        end
        
        #Generate elementary reflector H(i) to annihilate A(i+2:n,i)
        
        v,stau=householder_reflector!(A[i+1:n,i],V[i:n-1],n-i)
        A[i+1,i] -= stau*v[1]*dot(v,A[i+1:end,i])
        A[i+2:end,i] = v[2:end]
        tau[i]=stau
        E[i]=A[i+1,i]
        A[i+1,i]=1
        #Compute W[i+1:n,i]
        
        #skewmv(A[i+1:n,i+1:n], A[i+1:n,i],W[i+1:n,i],n-i)
        #LinearAlgebra.BLAS.gemv!('n',1.0,A[i+1:n,i+1:n], A[i+1:n,i],0.0,W[i+1:n,i])
        mul!(W[i+1:n,i],A[i+1:n,i+1:n], A[i+1:n,i])
        if i>1
            mul!(W[1:i-1,i],transpose(W[i+1:n,1:i-1]),A[i+1:n,i])
            #LinearAlgebra.BLAS.gemv!('T',1.0,W[i+1:n,1:i-1],A[i+1:n,i],0.0,W[1:i-1,i])
            #W[i+1:n,i] += A[i+1:n,1:i-1]*W[1:i-1,i]
            LinearAlgebra.BLAS.gemv!('n',1.0,A[i+1:n,1:i-1],W[1:i-1,i],1.0,W[i+1:n,i])
            mul!(W[1:i-1,i],transpose(A[i+1:n,1:i-1]),A[i+1:n,i])
            #LinearAlgebra.BLAS.gemv!('T',1.0,A[i+1:n,1:i-1],A[i+1:n,i],0.0,W[1:i-1,i])
            #W[i+1:n,i] -= W[i+1:n,1:i-1]*W[1:i-1,i]
            LinearAlgebra.BLAS.gemv!('n',-1.0,W[i+1:n,1:i-1],W[1:i-1,i],1.0,W[i+1:n,i])
        end
        
        W[i+1:n,i].*=stau
        alpha=-0.5*stau*dot(W[i+1:n,i],A[i+1:n,i])
        axpy!(alpha,A[i+1:n,i],W[i+1:n,i])
        
    end
    return 
end
function set_nb(n)
    if n<=12
        return max(n-3,1)
    elseif n<=100
        return 10
    else
        return 20
    end
    return 1
end

@views function sktrd!(A,n)
    nb=set_nb(n)

    E=similar(A,n-1)
    tau=similar(A,n-1)
    W=similar(A,n,nb)
    K=similar(A,n-nb,n-nb)
    V=similar(A,n-1)
    oldi=0
    @inbounds for i=1:nb:n-nb-2
        size=n-i+1
        
        latrd!(A[i:n,i:n],E[i:end],W,V,tau[i:end],size,nb)
        
        mul!(K[1:n-nb-i+1,1:n-nb-i+1],A[i+nb:n,i:i+nb-1],transpose(W[nb+1:size,:]))
        s=i+nb-1
        @inbounds for j=1:n-s
            for k=1:j-1
                @inbounds A[s+j,s+k] += K[j,k]-K[k,j]
                A[s+k,s+j] =- A[s+j,s+k]
            end
            @inbounds A[s+j,s+j]=0
        end
        for j=i:i+nb-1
            @inbounds A[j+1,j]=E[j]
        end
        oldi=i
    end
    oldi+=nb
    if oldi<n
        skewhess!(A[oldi:n,oldi:n])
    end
    return 
    
end
