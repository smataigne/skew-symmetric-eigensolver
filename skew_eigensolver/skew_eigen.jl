using LinearAlgebra

include("householder.jl")
include("skew_block_hess.jl")
include("round.jl")

@views function eig_of_skew_block(A,val)
    tr=A[1,1]+A[2,2]
    det=A[1,1]*A[2,2]-A[1,2]*A[2,1]
    rho=tr*tr-4*det
    if rho>=0
        val[1]=0
        val[2]=0
    else
        srho=sqrt(-rho)
        val[1]=(tr-1im*srho)/2.0
        val[2]=(tr+1im*srho)/2.0
    end
    return

end 
@views function doubleShiftcoeff(A,eig2)
    eig_of_skew_block(A,eig2)
    if isreal(eig2)
        if abs(eig2[1]-A[end,end])<abs(eig2[2]-A[end,end])
            c1=-2.0*eig2[1]
            c2=eig2[1]^2
        else
            c1=-2.0*eig2[2]
            c2=eig2[2]^2
        end
    else
        c1=-2.0*real(eig2[1])
        c2=abs2(eig2[1])
    end
    return c1,c2
end

@views function implicit_step(H,c1,c2,s,x,v,n)
    for i=1:n-2
        h=min(n,i+3)
        l=max(1,i-1)
        x[1]=dot(H[i,l:h],H[l:h,i])+c1*H[i,i]+c2
        x[2]=0
        x[3]=dot(H[i+2,l:h],H[l:h,i])+c1*H[i,i+2]

        v,tau = householder_reflector!(x,v,3)
        """
        if h-l==4
            s[l]  = dot(H[i:i+2,l],v)
            s[l+2]= dot(H[i:i+2,l+2],v)
            s[h]  = dot(H[i:i+2,h],v)
            for t1=i:2:i+2
                t3=t1-i+1
                for t2=l:2:h
                    H[t1,t2] -= tau*v[t3]*s[t2]
                end
            end
            s[l]  = dot(H[l,i:i+2],v)
            s[l+2]= dot(H[l+2,i:i+2],v)
            s[h]  = dot(H[h,i:i+2],v)
            for t1=i:2:i+2
                t3=t1-i+1
                for t2=l:2:h
                    H[t2,t1] -= tau*v[t3]*s[t2]
                end
            end
        else
            if l!=i-1
                l2=l+1
            else
                l2=l
            end
            s[l2]= dot(H[i:i+2,l2],v)
            s[l2+2]  = dot(H[i:i+2,l2+2],v)

            for t1=i:2:i+2
                t3=t1-i+1
                for t2=l2:2:l2+2
                    H[t1,t2] -= tau*v[t3]*s[t2]
                end
            end

            s[l2]=dot(H[l2,i:i+2],v)
            s[l2+2]=dot(H[l2+2,i:i+2],v)

            for t1=i:2:i+2
                t3=t1-i+1
                for t2=l2:2:l2+2
                    H[t2,t1] -= tau*v[t3]*s[t2]
                end
            end
        end
        
        #display(H)
        
        """
        leftHouseholder!(H[i:i+2,l:h],v,s[l:h],tau)
        
        rightHouseholder!(H[l:h,i:i+2],v,s[l:h],tau)
        
        
        
        
    end
    return

end

@views function QR_with_shifts(A)
    n=size(A,1)
    #H=zeros(n,n)
    s=similar(A,n)
    v=similar(A,3)
    x=similar(v,3)
    eig2=zeros(2)*1im
    sktrd!(A)
    
    #A=Matrix(hessenberg(A).H)
    """
    for i=1:n-1
        H[i+1,i]=A[i+1,i]
        H[i,i+1]=-H[i+1,i]
    end
    """
    for i=1:n-1
        A[i,i+1]=-A[i+1,i]
    end
    for i=3:n
        for j=1:i-2
            A[i,j]=0
            A[j,i]=0
        end
    end
    H=A
    
    tol=1e-8*norm(H)
    n_iter=16*n
    count_iter=0
    n_converged=0
    values=zeros(n)+im*zeros(n)
    N=n

    while n_converged<N && count_iter<=n_iter

        (c1,c2)=doubleShiftcoeff(H[n-1:n,n-1:n],eig2)
        implicit_step(H[1:n,1:n],c1,c2,s[1:n],x,v,n)

        if abs(H[n,n-1])<tol
            n_converged+=1
            values[n_converged]=H[n,n]
            n-=1
        elseif abs(H[n-1,n-2])<tol
            n_converged+=2
            eig_of_skew_block( H[n-1:n,n-1:n],values[n_converged-1:n_converged] )
            n-=2
        end

        if n==1
            values[end]=H[1,1]
            return values
        elseif n==2
            eig_of_skew_block(H[1:2,1:2],values[end-1:end])
            return values
        end
        count_iter+=1
        
    end   
end
@views function WYform(A,tau)
    n=size(A,1)
    WY=zeros(n-1,n-1)
    W=zeros(n-1,n-2)
    Yt=zeros(n-2,n-1)
    temp=zeros(n-1)
    
    for i=1:n-2
        t=tau[i]
        v=A[i+1:n,i]
        v[1]=1
        if i==1
            W[i:end,i]=-t*v
            Yt[i,i:end]=v
        else
            mul!(temp[1:i-1],Yt[1:i-1,i:end],v)
            Yt[i,i:end]=v
            W[i:end,i]=v
            LinearAlgebra.BLAS.gemv!('n',1.0,W[:,1:i-1],temp[1:i-1],1.0,W[:,i])
            W[:,i].*= -t
        end
    end
    
    display(W)
    mul!(WY,W,Yt)
    return WY
end

@views function skewhesseig(A)
    n=size(A,1)
    tau=sktrd!(A)
    s=similar(A,n)
    v=similar(A,n-1)
    for j=1:n-1
        @inbounds v[j] = A[j+1,j]
    end

    H=SymTridiagonal(zeros(n),v)
    trisol=eigen(H)
    
    vals  = trisol.values
    Qdiag = trisol.vectors
    Qr   = similar(A,(n+1)÷2,n)
    Qim  = similar(A,n÷2,n)
    temp =similar(A,n,n)
    
    """
    Qtri=similar(A,n,n)*1im
    c=1
    for i=1:n
        Qtri[i,:]=Qdiag[i,:].*c
        c=c*1im
    end
    for i=1:n-2
        t=tau[n-i-1]
        v=A[n-i:n,n-i-1]
        v[1]=1
        Qtri[n-i:n,:] -= t*v*transpose(v)*Qtri[n-i:n,:]

    end
    """
    
    
    
    #WYt=WY(A,tau)
    """
    S=copy(Qr[2:n,2:n])
    LinearAlgebra.BLAS.gemm!('n','n',1.0,WYt,S,1.0,Qr[2:n,2:n])
    S=copy(Qim[2:n,2:n])
    LinearAlgebra.BLAS.gemm!('n','n',1.0,WYt,S,1.0,Qim[2:n,2:n])
    LinearAlgebra.BLAS.gemv!('n',1.0,W,Qr[2:n,1],1.0,Qr[2:n,1])
    LinearAlgebra.BLAS.gemv!('n',1.0,W,Qim[2:n,1],1.0,Qim[2:n,1])
    """
    
    Q=Matrix(diagm(ones(n)))
    Q1=zeros(n,(n+1)÷2)
    Q2=zeros(n,n÷2)
    
    for i=1:n-2
        t=tau[n-i-1]
        v[1:i+1]=A[n-i:n,n-i-1]
        v[1]=1
        leftHouseholder!(Q[n-i:n,n-i-1:n],v[1:i+1],s[n-i-1:n],t)
        
    end
    
    c=1
    #vec=similar(A,n)
    @inbounds for i=1:2:n-1
        k1=(i+1)÷2
        Qr[k1,:] = Qdiag[i,:]
        Qim[k1,:] = Qdiag[i+1,:]
        Qr[k1,:].*=c
        Qim[k1,:].*=c
        #vec=Q[:,(i+1)÷2]
        Q1[:,(i+1)÷2] = Q[:,i]
        #Q[:,i] = vec
        Q2[:,(i+1)÷2] = Q[:,i+1]
        c*=(-1)
    end
    if n%2==1
        Qr[(n+1)÷2,:] = Qdiag[n,:]
        Qr[(n+1)÷2,:].*=c
        Q1[:,(n+1)÷2] = Q[:,n]
    end
    
    
    """
    Qr=LinearAlgebra.BLAS.gemm('n','n',1.0,Q,Qr)
    Qim=LinearAlgebra.BLAS.gemm('n','n',1.0,Q,Qim)
    """
    """
    WYform(A,tau)
    WY=zeros(n,n)
    Yt=zeros(n-2,n)
    for i=1:n-2
        Yt[i,i+1:end]=A[i+1:n,i]
        Yt[i,i+1]=1
    end
    display(W[:,1:oldi-1])
    display(Yt[1:oldi-1,:])
    mul!(WY,W[:,1:oldi-1],Yt[1:oldi-1,:])
    R=I-WY
    display(R)
    display(oldi)
    for i=oldi:n-2
        t=tau[i]
        v=A[i+1:n,i]
        v[1]=1
        #display(Q[n-i:n,n-i-1:n])
        rightHouseholder!(R[i+1:n,:],v,s,t)
        
    end
    display(R)
    display(Q)
    """
    
    mul!(temp,Q1,Qr) #temp is Qr
    mul!(Qdiag,Q2,Qim) #Qdiag is Qim
    
    return vals,temp,Qdiag

    
end

function main()
    BLAS.set_num_threads(1)
    n=1000
    A = randn(n,n)#zeros(n,n)
    for i=1:n
        A[i,i]=0
        for j=i+1:n
            #A[i,j]=i+j-1
            A[j,i]=-A[i,j]
        end
    end
    
    B = randn(n,n)
    for i=1:n
        B[i,i]=0
        for j=i+1:n
            B[j,i]=B[i,j]
        end
    end
    B=Symmetric(B)
    
    #display(hessenberg(A).H)
    #display(eigen(A).values)
    #display(eigen(A).vectors)
    @btime eigen(B) setup=(B = copy($B))
    @btime skewhesseig(A) setup=(A = copy($A))
    #val,Qr,Qim=skewhesseig(A)
    
    #display(values)
    """
    display(vectors)
    """
    #display(Qr)
    #display(Qim)
    
    #display(RoundMatrix(vectors*adjoint(vectors),1))
   
    
    #display((B*vectors[:,1])./values[1])
    #display(values)

    #display(eigvals(A))
    #@btime eigvals(B) setup=(B = copy($B))
    #@btime eigvals(H) setup=(H = copy($H))
    #@btime values= QR_with_shifts(A) setup=(A = copy($A))
    #@btime hessenberg(B).H setup=(B = copy($B))
    #@btime sktrd!(A) setup=(A = copy($A))
    a=1
    
end
#main()