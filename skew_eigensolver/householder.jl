using LinearAlgebra

@views function householder_reflector!(x,v,n)
    v[2:n]=x[2:n]
    v .*= 1/(x[1]+sign(x[1])*norm(x))
    v[1]=1
    tau = 2/((norm(v)^2))
    return v,tau
end
@views function leftHouseholder!(A,v,s,tau)
    LinearAlgebra.BLAS.gemv!('T',1.0,A,v,0.0,s)
    LinearAlgebra.BLAS.ger!(-tau,v,s,A)
    return
end
@views function rightHouseholder!(A,v,s,tau)
    LinearAlgebra.BLAS.gemv!('N',1.0,A,v,0.0,s)
    LinearAlgebra.BLAS.ger!(-tau,s,v,A)
    return 
end