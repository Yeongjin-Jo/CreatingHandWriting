# 주제 : 손글씨 만들어드립니다. 
## 부제 : 나만의 손글씨 생성 프로젝트

![title](https://user-images.githubusercontent.com/66348480/169939908-043fa8e3-4e7a-43d9-890f-57507fc4364e.PNG)

#### GAN을 활용해서 상용한글 2350자를 만들어내고 최종적으로 TTF 폰트파일까지 만들어줍니다.

## 사용법
### Download pretrained model
- 먼저 pretrained 모델을 받아줍니다(1800_D,G.pth). 이 모델은 약 30개의 중국 폰트와 30개의 한글폰트가 사전에 학습된 모델입니다. 
### Writing Template
- 사용자가 템플릿에 399글자를 씁니다. 순서대로 ```1-uniform.png```, ```2-uniform.png```, ```3-uniform.png``` 으로 이름을 설정해줍니다. src_dir 폴더에 넣습니다.
### Preprocessing Data
- Cropping : template을 일정한 규격대로 잘라줍니다. 
```sh 
python crop.py --src_dir=src_dir
               --dst_dir=dst_imgs
```
- font2img : base폰트와 함께 pair 이미지로 만들어줍니다. 
```sh- 
python font2img.py --src_font=NanumGothic.ttf 
                   --dst_imgs=dst_imgs
                   --charset=char399.txt
                   --sample_count=399 
                   --sample_dir=sample_dir
                   --label=label
                   --shuffle 
                   --filter 
                   --mode=hangeol_font2imgs
```
- package : 라벨을 포함하여 3차원 train 데이터와 val 데이터로 나누어줍니다. 
```sh
python package.py --dir=sample_dir
                  --save_dir=binary_save_directory
                  --split_ratio=0.1
```
### TransferLearning
- pretrained model에 사용자의 글씨를 학습시킵니다. 
```sh
python train.py --experiment_dir=experiment_dir
                --binary_save_directory=binary_save_directory
                --resume 42000
                --gpu_ids=cuda:0
                --input_nc=1
                --batch_size=batch_size
                --epoch=epoch
                --sample_steps=200
                --checkpoint_steps=1000
```
### inference
- 학습시킨 모델을 이용해 나머지 한글을 생성합니다. 
```sh
python infer.py --experiment_dir=experiment_dir
                --gpu_ids cuda:0
                --batch_size batchsize
                --resume modelname
                --src_font NanumGothic.ttf 
                --from_2350 
                --label
```
### MakeSVG
- 나온 한글 2350자 png를 ubuntu 환경에서 svg로 변환합니다. 
```sh
$ sh png2pnm2svg.sh
```
- ** 만일, access denied 된다면, ```sudo chmod 775```로 권한을 부여해줍니다. 
### MakeTTF
- Fontforge python script를 이용하여, 사용자의 글자를 폰트 파일로 변환합니다. 
```sh
python3 MakeTTF.py --src_font=NanumGothic.ttf
                   --png_path=png_path 
                   --new_font=newfontname
```
## 도움 준 자료들 
- [zi2zi](https://github.com/kaonashi-tyc/zi2zi) by [kaonashi-tyc](https://github.com/kaonashi-tyc)
- [EuphoriaYan
/
zi2zi-pytorch](https://github.com/EuphoriaYan/zi2zi-pytorch)
- [periannath
/
neural-fonts](https://github.com/periannath/neural-fonts)
- [jeina7
/
GAN-handwriting-styler](https://github.com/jeina7/GAN-handwriting-styler)

* 발표 자료 ppt가 따로 있습니다. 
