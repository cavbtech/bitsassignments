
for i = 1:size(finalM,1)
    eigVal = double(eig(i))
    if(eigVal)==0
        disp(eigVal + ' - critical point is degenerate')
        if(eigVal >0 )
            disp(eigVal + ' - Critical point is minimum')
        else
            disp(eigVal + ' - Cricical point is maximum')
        end
    else
        disp(eigVal + ' - Critical point is saddle')
    end
end