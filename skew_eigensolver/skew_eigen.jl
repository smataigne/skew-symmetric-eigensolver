using LinearAlgebra
using PyPlot
include("householder.jl")
include("skew_blok_hess.jl")
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
    sktrd!(A,n)
    
    #A=hessenberg(A).H
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




function main()
    n=10
    A = zeros(n,n)
    for i=1:n
        for j=i+1:n
            A[i,j]=i+j-1
            A[j,i]=-A[i,j]
        end
    end
    
    #display(eigvals(A))
    #@btime eigvals(A) setup=(A = copy($A))
    #@btime values= QR_with_shifts(A) setup=(A = copy($A))
    a=1
    display(eigvals(A))
    values= QR_with_shifts(A) 
    display(values)
    
end
main()