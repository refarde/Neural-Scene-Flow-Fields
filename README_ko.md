# Neural Scene Flow Fields
"Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes", CVPR 2021 논문의 pytorch 구현의 pork

[[Project Website]](https://www.cs.cornell.edu/~zl548/NSFF/) [[Paper]](https://arxiv.org/abs/2011.13084) [[Video]](https://www.youtube.com/watch?v=qsMIH7gYRCc&feature=emb_title)

[[Re-implementation and imrpovement of NSFF]](https://github.com/kwea123/nsff_pl)

## 들어가기 전에
해당 코드는 다음 환경에서 테스트함

| | |
|-|-|
|OS|Windows10|
|Shell|bash (git bash)|

## 필요 모듈 설치
### pytorch
```bash
$ pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```
### cupy
```bash
$ pip3 install --upgrade cupy-cuda111
```
### 나머지
```bash
$ pip3 install configargparse matplotlib scikit-image scipy imageio tqdm --upgrade
$ pip3 install tensorboard --upgrade
```

## 전처리
1. 테스트에 사용할 샘플이미지 준비.   
   만약 동영상에서 프레임을 추출해 사용한다면 아래 명령 실행
    ```bash
    # 아래 명령 실행시, nsff_data/dense/images 폴더 안에 동영상의 프레임이 6 프레임 간격으로 최대 30개의 파일이 추출됨.
    $ python mp42png.py --input_path /Path/of/video.mp4
    ```
2. [COLMAP](https://demuc.de/colmap/)을 이용하여 sparse 데이터 추출
   1. UI를 사용할 경우
      1. colmap 앱을 실행 실행
      2. File > New Project로 프로젝트 생성
      3. Processing > Feature extraction 실행 후,   
         Camera model을 PINHOLE로 변경 후에
         Extract 클릭
      4. Processing > Feature matching 실행 후,
         Exhaustive 탭에서 Run 클릭
      5. Reconstruction > Start reconstruction 실행
      6. File > Extract model 실행 후   
         생성된 파일(camera.bin 등)을 nsff_data/dense/sparse 폴더로 이동
   2. cli 를 사용할 경우
      ```bash
      $ ../COLMAP-3.6-windows-cuda/colmap.bat feature_extractor --database_path ./nsff_data/database.db --image_path ./nsff_data/images --ImageReader.camera_model PINHOLE --ImageReader.single_camera 1 --ImageReader.mask_path ./nsff_data/colmap_masks

      $ ../COLMAP-3.6-windows-cuda/colmap.bat sequential_matcher --database_path ./nsff_data/database.db

      $ mkdir ./nsff_data/sparse

      $ ../COLMAP-3.6-windows-cuda/colmap.bat mapper --database_path ./nsff_data/database.db --image_path ./nsff_data/images --output_path ./nsff_data/sparse

      $ mkdir ./nsff_data/dense

      $ ../COLMAP-3.6-windows-cuda/colmap.bat image_undistorter --image_path ./nsff_data/images --input_path ./nsff_data/sparse/0 --output_path ./nsff_data/dense --output_type COLMAP
      ```

3. 데이터 전처리   
   1. Single view depth prediction model 파일인 "model.pt"을 [link](https://drive.google.com/drive/folders/1G-NFZKEA8KSWojUKecpJPVoq5XCjBLOV?usp=sharing)로 부터 다운로드 받고, "nsff_scripts" 폴더 안에 넣음.

   2. 다음 명령 실행:
      ```bash
      $ cd nsff_scripts
      # create camera intrinsics/extrinsic format for NSFF, same as original NeRF where it uses imgs2poses.py script from the LLFF code: https://github.com/Fyusion/LLFF/blob/master/imgs2poses.py
      $ python save_poses_nerf.py --data_path "../nsff_data/dense/"
      # Resize input images and run single view model
      $ python run_midas.py --data_path "../nsff_data/dense/" --input_w 640 --input_h 360 --resize_height 288
      # Run optical flow model (for easy setup and Pytorch version consistency, we use RAFT as backbond optical flow model, but should be easy to change to other models such as PWC-Net or FlowNet2.0)
      $ ./download_models.sh
      $ python run_flows_video.py --model models/raft-things.pth --data_path ../nsff_data/dense/ --epi_threshold 1.0 --input_flow_w 768 --input_semantic_w 1024 --input_semantic_h 576
      ```

## 학습
```bash
cd nsff_exp
# configs/config_nsff_data.txt의 하이퍼 파라미터를 다음과 같이 수정 후 학습 진행
# (상세 하이퍼파라미터는 readme 참조)
#   datadir = ../nsff_data/dense
#   end_frame = 마지막 생성된 파일 인덱스
python run_nerf.py --config configs/config_nsff_data.txt
```

## Render
### nsff_exp/config.txt와 args.txt의 datadir를 현재 경로로 수정
```bash
cd nsff_exp
# 공간만 보간
python run_nerf.py --config configs/config_nsff_data.txt --render_bt --target_idx 10
# 시간만 보간
python run_nerf.py --config configs/config_nsff_data.txt --render_lockcam_slowmo --target_idx 10
# 시간/공간 보간
python run_nerf.py --config configs/config_nsff_data.txt --render_slowmo_bt --target_idx 10
```

## License
This repository is released under the [MIT license](hhttps://opensource.org/licenses/MIT).

## Citation
If you find our code/models useful, please consider citing our paper:
```bash
@InProceedings{li2020neural,
  title={Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes},
  author={Li, Zhengqi and Niklaus, Simon and Snavely, Noah and Wang, Oliver},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2021}
}
