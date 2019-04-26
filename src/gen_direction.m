dirPath='F:/[A]project/hairMatting/matting/';
for f_num = [6 43 111]
%     fname=[num2str(f_num,'%05d') '.png'];
    fname=[num2str(f_num,'%d') '_rgb.png'];
    img=imread([dirPath 'dataset/test2/rgb/' fname]);
    alpha=imread([dirPath 'dataset/test2/alpha/' fname]);
    for i=1:size(alpha,1)
        for j=1:size(alpha,2)
            if alpha(i,j)==0
                img(i,j,1)=255;
                img(i,j,2)=0;
                img(i,j,3)=0;
            end
        end
    end
    % figure;
    % imshow(img);
    return_map = generate_dataX(img,alpha);
    eval(['dir' fname(1:5) '= return_map;']);
    save([dirPath 'dataset/test2/direction/' fname '.mat'],['dir' fname(1:5)],'-v7.3','-nocompression') 
%     clear dir*
end