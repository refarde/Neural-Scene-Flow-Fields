import glob
from PIL import Image

# filepaths
base_dir = "./nsff_exp/logs/kid-running_ndc_5f_sv_of_sm_unify3_3_F00-24/"

tar_num = "280001"

fps = [{
    "fp_in": base_dir + "images_512x288/*.png",
    "fp_out": base_dir + "images_512x288.gif"
}, {
    "fp_in": base_dir + "render-lockcam-slowmo_" + tar_num + "/*.jpg",
    "fp_out": base_dir + "render-lockcam-slowmo_" + tar_num + ".gif"
}, {
    "fp_in": base_dir + "render-slowmo_bt_path_" + tar_num + "/images/*.jpg",
    "fp_out": base_dir + "render-slowmo_bt_path_" + tar_num + ".gif"
}, {
    "fp_in": base_dir + "render-spiral-frame-010_path_" + tar_num + "/images/*.jpg",
    "fp_out": base_dir + "render-spiral-frame-010_path_" + tar_num + ".gif"
}]

for i in fps:
    print(i)
    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    img_list = glob.glob(i["fp_in"])
    if len(img_list) > 0 :
        img, *imgs = [Image.open(f) for f in sorted(img_list)]
        img.save(fp=i["fp_out"], format='GIF', append_images=imgs,
                save_all=True, duration=6, loop=0)
