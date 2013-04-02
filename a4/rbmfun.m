function [hidbiases, vishid] = rbmfun(data,numhid,weightcost,maxepoch)

epsilonw      = 0.05;
epsilonvb     = 0.05;
epsilonhb     = 0.05;

initialmomentum  = 0.5;
finalmomentum    = 0.9;


  [numcases numdims]=size(data);

  epoch=1;
%  poshidprobs = zeros(numcases,numhid);
%  neghidprobs = zeros(numcases,numhid);
%  posprods    = zeros(numdims,numhid);
%  negprods    = zeros(numdims,numhid);
  vishid     = 0.03*randn(numdims, numhid);
  hidbiases  = 0*ones(1,numhid);
  visbiases  = zeros(1,numdims);
  vishidinc  = zeros(numdims,numhid);
  hidbiasinc = zeros(1,numhid);
  visbiasinc = zeros(1,numdims);

for epoch = epoch:maxepoch,
  poshidprobs = 1./(1 + exp(-data*vishid - repmat(hidbiases,numcases,1)));    
  posprods    = data' * poshidprobs;
  poshidact   = sum(poshidprobs);
  posvisact = sum(data);

%%%%%%%%% END OF POSITIVE PHASE  %%%%%%%

poshidstates = poshidprobs > rand(numcases,numhid);

%%%%%%%%  START NEGATIVE PHASE  %%%%%%%%%

  negdata = 1./(1 + exp(-poshidstates*vishid' - repmat(visbiases,numcases,1)));
  neghidprobs = 1./(1 + exp(-negdata*vishid - repmat(hidbiases,numcases,1)));    
  negprods  = negdata'*neghidprobs;
  neghidact = sum(neghidprobs);
  negvisact = sum(negdata); 

%%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%

  errsum= sum(sum( (data-negdata).^2 ));
   if epoch>5,
     momentum=finalmomentum;
   else
     momentum=initialmomentum;
   end;

    vishidinc = momentum*vishidinc + ...
                epsilonw*( (posprods-negprods)/numcases - weightcost*vishid);
    visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact);
    hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact);
    vishid = vishid + vishidinc;
    visbiases = visbiases + visbiasinc;
    hidbiases = hidbiases + hidbiasinc;


    if rem(epoch,10) ==0,
      fprintf(1, 'numhid %4.0i epoch %4.0i  reconstruction error %6.1f  \n', numhid, epoch, errsum); 
    end;
   
end;