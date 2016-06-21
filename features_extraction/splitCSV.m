file_path = '/media/HDD_2TB/DATASETS/MSVD/train_C3D_ImageNet_features.csv';

[path, name, format] = fileparts(file_path);

f = fopen(file_path, 'r');
f_ = fopen(sprintf([path '/' name '_part_%0.6d.' format], 1), 'w');
i = 0; nfiles = 1; finished = false;

while(~finished) 
	while(~finished && i < 10000)
		line = fgets(f);
		if(~ischar(line))
			finished = true;
		else
			fprintf(f_, line);
			i = i+1;
		end
	end
	fclose(f_);
	if(~finished)
		nfiles = nfiles+1; 
		f_ = fopen(sprintf([path '/' name '_part_%0.6d.' format], nfiles), 'w');
	end
	i = 0;
end
fclose(f_);
fclose(f);
