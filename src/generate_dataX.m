function return_map = generate_dataX(img,depth)
%   Input: RGB image , Depth map(1 channel)
%   Return: image of 4 channels. (wang: change to 3 channel)
%                 DirectX, DirectY, Confidence map(norm), depth map(norm).
    I=img;
    I=rgb2gray(I);
    iter_num=3;
    w_normal=I;%confidence map.
    %figure; imshow(depth);
    for it=1:iter_num
        %gabor filter 
        gaborBank = gabor(5,0:5.625:180-5.625);
        gaborMag = imgaborfilt(w_normal,gaborBank);
        [gabor_max,gabor_id]=max(gaborMag,[],3);
        
        %caculate the confidence map
        w=zeros(size(gabor_id));
        for i=1:32
            dis_map=dis_angle(5.625*(i-1),5.625*gabor_id);
            w_tmp=(dis_map.*(gaborMag(:,:,i)-gabor_max).^2).^0.5;
            w=w_tmp+w;
        end
        % threshold our confidence map.
        w_max=max(max(w));
        w_threshold=(w>w_max*0.01);

        w_normal=w.*double(w_threshold);
        
        %normal the confidence map
        w_normal=norm(w_normal,w_threshold);
        
        %show the color map.
        %figure; imshow(to_hsv(w_normal));
    end
    %  get the direction map X and Y
    confidence_map=w_normal;
    direction_map=(gabor_id-1)*5.625;
    direction_X=cos(direction_map/180*pi);
    direction_Y=sin(direction_map/180*pi);
    direction_X=double(w_threshold).*direction_X;
    direction_Y=double(w_threshold).*direction_Y;
    %figure; imshow(to_hsv(direction_X));
    %figure; imshow(to_hsv(direction_Y));
    size_d=size(direction_X);
%     return_map=zeros(size_d(1,1),size_d(1,2),4);
    return_map=zeros(size_d(1,1),size_d(1,2),3);
    return_map(:,:,1)=direction_X;
    return_map(:,:,2)=direction_Y;
    return_map(:,:,3)=confidence_map;
%     return_map(:,:,4)=1-double(depth)/255;
    
    %-------------------------------------------------------
    function p_norm=norm(img,mask)
        %img only have 1 channel, and mask is a bool map.
            img_mask_remove=img.*double(mask);
            img_max=max(max(img_mask_remove));
            img_temp=double(~mask)*(img_max+1)+img_mask_remove;
            img_min=min(min(img_temp));
            p_norm=double(mask).*(img_mask_remove-img_min)./(img_max-img_min);
    end

    %------------------------------------------------------------
    function d=dis_angle(a,b)
            %this function is to calculate the angle distance .
            d=min(abs(a-b),abs(a-b+180));
            d=min(d,abs(a-b-180));
            d=d/180*pi;
    end
    %-------------------------------------------------------
    function hsv=to_hsv(img)
            % this function is to show the image(0-1) to color map.
            size_img=size(img);
            hsv_img=zeros(size_img(1,1),size_img(1,2),3);
            hsv_img(:,:,1)=img;
            hsv_img(:,:,2)=1;
            hsv_img(:,:,3)=1;
            hsv=hsv2rgb(hsv_img);
    end
end

