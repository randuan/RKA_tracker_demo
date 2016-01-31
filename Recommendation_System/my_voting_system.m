% algorithm 2: VotingProcess
% input: G, alpha
function out = my_voting_system(G,alpha)

if nargin < 2
    alpha = 0.85;
end

n = size(G,1);

cj=sum(G,1);% sum of column of G
ci=sum(G,2);

D=diag(1./cj);
B=diag(1./ci);

A=alpha*G*D+(1-alpha)*B*G;

x=ones(n,1);% init vector x

cnt=1;% recording the times of iteration
while cnt < n
    x=A*x;
    cnt=cnt+1;
    x(x < 1) = 1;
end

[x1,index]=sort(x);%ranking the x
x1=flipud(x1);%ranking the rows of x1 from max to min
index=flipud(index);
%out put the result
out=[1:n;x1';index'];

return
