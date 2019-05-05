maindir = '../dataset/blender_render_results/imgs/';
subdir=dir(maindir);

savedir='../dataset/train2/';

cnt = 1;
for i=1:length(subdir)
    if(isequal(subdir(i).name,'.')||isequal(subdir(i).name,'..')||~subdir(i).isdir)
        continue;
    end
    subdirpath = fullfile( maindir, subdir( i ).name, 'hair*' );
    images = dir( subdirpath );
    for j = 1 : length( images )
        imagepath = fullfile( maindir, subdir( i ).name, images( j ).name  );
        [I, map, Transparency] = imread( imagepath ); 
        imwrite(I,[savedir 'rgb/' num2str(cnt,'%05d') '.png']);
        imwrite(Transparency,[savedir 'alpha/' num2str(cnt,'%05d') '.png']);
        cnt=cnt+1;
    end
end