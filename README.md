Origin - main

Steps:
1. Clone the repository:
  - git clone https://github.com/nishi-v/clothes_virtual_tryon.git


2. Go to directory IDM-VTON

   Run the download_models.sh file which is present in root folder.
   - cd IDM-VTON

   run download_model.sh
   - bash download_model.sh

4. Create a folder named yisol in IDM-VTON
   - mkdir yisol
   - cd yisol

   Clone the below mentioned repository there:
   - apt-get install lfs
   - git lfs install
   - git clone https://huggingface.co/yisol/IDM-VTON
  
5. Finally run gradient.ipynb file
   (come back to IDM-VTON folder)
   - cd ..
   - python3 gradio_demo/app.py

