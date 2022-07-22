using LinearAlgebra
include("skew_eigen.jl")

@views function skewexpm(A)
    n=size(A,1)
    vals,Qr,Qim = skewhesseig(A)
    
    temp1=similar(A,n,n)
    temp2=similar(A,n,n)
    QrS=copy(Qr)
    QrC=copy(Qr)
    QimS=copy(Qim)
    QimC=copy(Qim)
    for i=1:n
        c=cos(vals[i])
        s=sin(-vals[i])
        QrS[:,i].*=s
        QimS[:,i].*=s
        QrC[:,i].*=c
        QimC[:,i].*=c
    end

    mul!(temp1,QrC-QimS,transpose(Qr))
    mul!(temp2,QrS+QimC,transpose(Qim))
    return temp1+temp2
    """
    Q=Qr+Qim.*1im
    Q2=copy(Q)
    for i=1:n
        diag=cos(vals[i])+1im*sin(-vals[i])
        Q[:,i].*=diag
    end
    return Q*adjoint(Q2)
    """
end

BLAS.set_num_threads(1)
n=1000
A = randn(n,n)
for i=1:n
    A[i,i]=0
    for j=i+1:n
        #A[i,j]=i+j-1
        A[j,i]=-A[i,j]
    end
end

B=Symmetric(A)
@btime exp(A)

@btime skewexpm(A) setup=(A = copy($A))


"""
display(exp(A))
E=skewexpm(A)
display(E)
"""
a=1