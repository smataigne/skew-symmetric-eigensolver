function RoundVector(v,ndig)
    n=length(v)
    for i=1:n
            v[i]=round(v[i],digits=ndig)
    end
    return v
end
function RoundMatrix(A,ndig)
    n=length(A[1,:])
    for i=1:n
        for j=1:n
            A[i,j]=round(A[i,j],digits=ndig)
        end
    end
    return A
end