function [options] = defaultOptions(varargin)
    
    if mod(nargin,2) ~=0
        options=varargin{1};
        n=nargin-1;
        posp=0;
    else
        options=struct;
        n=nargin;
        posp=1;
    end
    n=floor(n/2);
    for i=1:n
       pos=2*i-posp;
       key=varargin{pos};
       if ~isfield(options,key)
           val=varargin{pos+1};
           options=setfield(options,key,val);
       end
    end
end


