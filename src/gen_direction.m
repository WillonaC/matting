for f_num = 1:300
    fname=[num2str(f_num,'%05d') '.png'];
    img=imread(['../dataset/test/rgb/' fname]);
    alpha=imread(['../dataset/test/alpha/' fname]);
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
    save(['./dataset/test/direction/' fname '.mat'],['dir' fname(1:5)],'-v7.3','-nocompression') 
    clear dir*
end