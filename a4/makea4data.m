% crate the file a4data.mat with the data to be used in A4

load data.mat

inputs = [data.training.inputs data.validation.inputs data.test.inputs];
targets = [data.training.targets data.validation.targets data.test.targets];

clear data

%a4data = struct('training',struct('inputs',[],'targets',[],'inputs_unlabelled',[])...
%    ,'validation',struct('inputs',[],'targets',[]),'test',struct('inputs',[],'targets',[]));

a4data = struct();

a4data.training.inputs = inputs(:,1:500);
a4data.training.targets = targets(:,1:500);
a4data.training.inputs_unlabelled = inputs(:,5001:11000);

a4data.validation.inputs = inputs(:,501:1000);
a4data.validation.targets = targets(:,501:1000);

a4data.test.inputs = inputs(:,1001:5000);
a4data.test.targets = targets(:,1001:5000);

clear inputs targets

save a4data.mat a4data

