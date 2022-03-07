clear all 
close all

X1=imread('cameraman.tif');

%%Bruit Gaussien
X2=imnoise(X1,'gaussian',0,0.005);
%Bruit impulsionnel
X3=imnoise(X1,'salt & pepper',0.05);

figure, subplot(1,3,1), imshow(X1), title('image originale')
subplot(1,3,2), imshow(X2), title('bruit Gaussien')
subplot(1,3,3), imshow(X3), title('bruit Sel&Poivre')

%Affichage ligne de l'image
figure, subplot(3,1,1), plot(X1(200,:)), title('image originale')
subplot(3,1,2), plot(X2(200,:)), title('bruit Gaussien')
subplot(3,1,3), plot(X3(200,:)), title('bruit Sel&Poivre')

%filtrage linéaire
%filtre moyenneur
h=fspecial('average',[3 3]);
Y2=imfilter(X2,h,'replicate');
h=fspecial('average',[5 5]);
Y22=imfilter(X2,h,'replicate');
figure, 
subplot(2,2,1), imshow(X1),title('image originale')
subplot(2,2,2), imshow(X2),title('bruit Gaussien')
subplot(2,2,3), imshow(Y2), title('filtre moyenneur 3x3')
subplot(2,2,4), imshow(Y22), title('filtre moyenneur 5x5')

%Filtrage Gaussien
h=fspecial('gaussian',[3 3],0.6);
Y222=imfilter(X2,h,'replicate');
figure, subplot(2,2,1), imshow(X1),title('image originale')
subplot(2,2,2), imshow(X2), title('bruit Gaussien, psnr=23,34')
subplot(2,2,3), imshow(Y2), title('filtre moyenneur 3x3, psnr=25,12')
subplot(2,2,4), imshow(Y222), title('filtre gaussien, psnr=27,03')

%ligne de l'image
figure, subplot(4,1,1), plot(X1(100,:)),title('image originale')
subplot(4,1,2), plot(X2(100,:)), title('bruit Gaussien, psnr=23,34')
subplot(4,1,3), plot(Y2(100,:)), title('filtre moyenneur 3x3, psnr=25,12')
subplot(4,1,4), plot(Y22(100,:)), title('filtre moyenneur 5x5')

psnr0=PSNRTP2(X1,X2)
psnr1=PSNRTP2(X1,Y2)
psnr2=PSNRTP2(X1,Y22)
psnr3=PSNRTP2(X1,Y222)

%filtrage non linéaire
Y3=imfilter(X3,h,'replicate');
Y33=medfilt2(X3,[3 3]);
figure, subplot(2,2,1), imshow(X1),title('image originale')
subplot(2,2,2), imshow(X3), title('bruit sel&poivre,psnr=18,21')
subplot(2,2,3), imshow(Y3),title('filtre gaussien, psnr=23,22')
subplot(2,2,4), imshow(Y33),title('filtre median, psnr=26,77')

figure, subplot(4,1,1), plot(X1(100,:)),title('image originale')
subplot(4,1,2), plot(X3(100,:)), title('bruit sel&poivre,psnr=18,21')
subplot(4,1,3), plot(Y3(100,:)), title('filtre gaussien, psnr=23,22')
subplot(4,1,4), plot(Y33(100,:)), title('filtre median, psnr=26,77')

psnr3=PSNRTP2(X1,X3)
psnr4=PSNRTP2(X1,Y3)
psnr5=PSNRTP2(X1,Y33)

