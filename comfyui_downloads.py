from pathlib import Path
bp = Path(__file__).parents[0]
bp = Path(bp).absolute()
venv_path = Path(bp) / 'installer_files/env'

from subprocess import check_call,  PIPE
from functools import partial
from shutil import rmtree
import sys, time, re, os, platform, stat

cc = partial(check_call, shell=True)

# Format: folder: [(hf_directory, file_name_1, file_name_2)]
hf_models = {
    'vae': [
        ('stabilityai/sd-vae-ft-mse-original', 'vae-ft-mse-840000-ema-pruned.ckpt'),
    #     # ("AIARTCHAN/aichan_blend", "vae/Berry's%20Mix.vae.safetensors"),
    ],
    'upscale_models': [
        ("uwg/upscaler", 
        #  "ESRGAN/8x_NMKD-Superscale_150000_G.pth",
         "ESRGAN/4x-UltraSharp.pth",
        #  "ESRGAN/UniversalUpscaler/4x_UniversalUpscalerV2-Neutral_115000_swaG.pth",
        #  "ESRGAN/UniversalUpscaler/4x_UniversalUpscalerV2-Sharp_101000_G.pth",
        #  "ESRGAN/UniversalUpscaler/4x_UniversalUpscalerV2-Sharper_103000_G.pth",
        ),
    ],
    'embeddings': [
        ("Lykon/DreamShaper", "UnrealisticDream.pt", "FastNegativeEmbedding.pt", "BadDream.pt"),
    ],
    'checkpoints': [
        ("Lykon/DreamShaper", "DreamShaper_8_pruned.safetensors", 
        #   'DreamShaper_8_INPAINTING.inpainting.safetensors'
        )
    ],
    bp / 'comfy-ui/custom_nodes/ComfyUI-AnimateDiff-Evolved/models': [
        ("guoyww/animatediff", # "mm_sd_v14.ckpt", "mm_sd_v15.ckpt", 
            # "mm_sd_v15_v2.ckpt",
            "v3_sd15_mm.ckpt"
            ),
        # ("CiaraRowles/TemporalDiff", "temporaldiff-v1-animatediff.ckpt"),
        # ("manshoety/AD_Stabilized_Motion", "mm-Stabilized_high.pth", "mm-Stabilized_mid.pth")
    ],
    # bp / 'ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved/motion_lora': [
    #     ("guoyww/animatediff", "v2_lora_PanDown.ckpt", "v2_lora_PanLeft.ckpt", "v2_lora_PanRight.ckpt", "v2_lora_PanUp.ckpt", "v2_lora_RollingAnticlockwise.ckpt",
    #      "v2_lora_RollingClockwise.ckpt", "v2_lora_ZoomIn.ckpt", "v2_lora_ZoomOut.ckpt")
    # ],
    'controlnet': [
        ('lllyasviel/ControlNet-v1-1',  
            'control_v11p_sd15_openpose.pth', 
            'control_v11f1p_sd15_depth.pth',
            # 'control_v11p_sd15_lineart.pth',  
            # 'control_v11p_sd15_canny.pth', 
            # 'control_v11f1e_sd15_tile.pth',
            # 'control_v11p_sd15_softedge.pth',  
        ),
        # ('CiaraRowles/TemporalNet', 'diff_control_sd15_temporalnet_fp16.safetensors')
    ],
    # # https://huggingface.co/TencentARC/T2I-Adapter/tree/main/models
    bp / 'comfy-ui/custom_nodes/ComfyUI_IPAdapter_plus/models': [
        ('h94/IP-Adapter', 
        #  'models/ip-adapter-plus-face_sd15.bin',
         'models/ip-adapter-plus_sd15.bin',
         'models/ip-adapter_sd15.bin',
        # 'models/ip-adapter_sd15_light.bin'
         )
    ],
    'clip_vision': [
        ('h94/IP-Adapter', 'models/image_encoder/model.safetensors')
    ],
    'facerestore_models': [
        ('nlightcho/gfpgan_v14', 'GFPGANv1.4.pth')
    ]
}

civit_models = {
    'checkpoints': [
        # '95489', # Any Lora
        # '167084', # AingDiffusion Anime
        # '197181'  # epiCPhotoGasm - Y Photoreal
    ],
    # Loras are much smaller files, so downloading extras makes sense
    'loras': [
        '87153',  '114197', "124161", "115255", "122112", "89697", "81702", "7657", "5397", "5713", "6503", "6872"
    ],
    bp / 'comfy-ui/custom_nodes/ComfyUI-AnimateDiff-Evolved/motion_lora': [
        '154110',
    ],
    # bp / 'comfy-ui/custom_nodes/ComfyUI-AnimateDiff-Evolved/models': [
    #     '178017', # Improved 3D Motion Module  https://civitai.com/api/download/models/178017
    # ],
}

misc_link_models = {
    'facerestore_models': [
        'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
    ],
    'upscalers': [
        'https://github.com/Koushik0901/Swift-SRGAN/releases/download/v0.1/swift_srgan_2x.pth.tar',
        'https://github.com/Koushik0901/Swift-SRGAN/releases/download/v0.1/swift_srgan_4x.pth.tar',
    ],
}

# Extensions
extensions = [
    ('ltdrdata', 'ComfyUI-Manager', 'ComfyUI-Impact-Pack', 'ComfyUI-Inspire-Pack'),
    ('Kosinkadink', 'ComfyUI-AnimateDiff-Evolved', 'ComfyUI-Advanced-ControlNet', 'ComfyUI-VideoHelperSuite'),
    ('Fannovel16', 'ComfyUI-Frame-Interpolation', 'comfyui_controlnet_aux', 'ComfyUI-MotionDiff'),
    ('pythongosssss', 'ComfyUI-Custom-Scripts'), # ComfyUI-WD14-Tagger (Last upated July 2023)
    ('WASasquatch', 'was-node-suite-comfyui', 'PowerNoiseSuite'),
    'SLAPaper/ComfyUI-Image-Selector',
    "Derfuu/Derfuu_ComfyUI_ModdedNodes",
    "cubiq/ComfyUI_IPAdapter_plus",
    'kijai/ComfyUI-KJNodes',
    "melMass/comfy_mtb",
    "RockOfFire/ComfyUI_Comfyroll_CustomNodes",
    "comfyanonymous/ComfyUI_experiments",
    'jamesWalker55/comfyui-various',
    'mav-rik/facerestore_cf',
    'jags111/efficiency-nodes-comfyui',
    # "Gourieff/comfyui-reactor-node",
    "FizzleDorf/ComfyUI_FizzNodes",
    "evanspearman/ComfyMath",
    "theUpsider/ComfyUI-Logic",
    "Ttl/ComfyUi_NNLatentUpscale",
    "JPS-GER/ComfyUI_JPS-Nodes",
]

repositories_without_lazy_install = [
    'comfy_mtb',
    'ComfyUI_FizzNodes',
    'ComfyUI-KJNodes',
    'facerestore_cf',
    'ComfyUI-MotionDiff',
    'comfyui-reactor-node',
    'ComfyUI-Impact-Pack',
    'was-node-suite-comfyui'
]

def pip_install(*args):
    # https://stackoverflow.com/questions/12332975/how-can-i-install-a-python-module-within-code
    print([sys.executable, "-m", "pip", "install", *args])
    check_call([sys.executable, "-m", "pip", "install", *args])

def git_clone(repository, dest_directory, skip_if_exists=True):
    # print(repository, not (skip_if_exists and Path(dest_directory).exists()))
    if not (skip_if_exists and Path(dest_directory).exists()):
        cc(f"git clone https://github.com/{repository} {Path(dest_directory).absolute().as_posix()}")

def download_files(url_dict, processes=10):
    """
    Downloads files asynchronously with asyncio.
    ::url_dict:: Dictionary of URLs to download
    ::processes:: The maximum amount of downloads to happen at once.
    This function returns some download stats that can be used with the print_stats function.
    
    
    Base Code from Source: https://gist.github.com/darwing1210/c9ff8e3af8ba832e38e6e6e347d9047a
    """

    def load_df_cm(bp):
        df_json = pd.read_json(Path(bp) / 'comfy-ui/custom_nodes/ComfyUI-Manager/model-list.json')
        df = pd.DataFrame.from_records(df_json.apply(lambda x:{**x['models']}, axis=1))
        df = pd.concat([df, df_json], axis=1).rename(columns={'models':'json'})
        return df

    def replace_dest(df, url):
        def get_model_dir(data):
            if data['save_path'] != 'default':
                if '..' in data['save_path'] or data['save_path'].startswith('/'):
                    print(f"[WARN] '{data['save_path']}' is not allowed path. So it will be saved into 'models/etc'.")
                    base_model = "etc"
                else:
                    if data['save_path'].startswith("custom_nodes"):
                        base_model = os.path.join(Path(bp) / 'comfy-ui', data['save_path'])
                    else:
                        base_model = os.path.join(folder_paths.models_dir, data['save_path'])
            else:
                model_type = data['type']
                if model_type == "checkpoints":
                    base_model = folder_paths.folder_names_and_paths["checkpoints"][0][0]
                elif model_type == "unclip":
                    base_model = folder_paths.folder_names_and_paths["checkpoints"][0][0]
                elif model_type == "VAE":
                    base_model = folder_paths.folder_names_and_paths["vae"][0][0]
                elif model_type == "lora":
                    base_model = folder_paths.folder_names_and_paths["loras"][0][0]
                elif model_type == "T2I-Adapter":
                    base_model = folder_paths.folder_names_and_paths["controlnet"][0][0]
                elif model_type == "T2I-Style":
                    base_model = folder_paths.folder_names_and_paths["controlnet"][0][0]
                elif model_type == "controlnet":
                    base_model = folder_paths.folder_names_and_paths["controlnet"][0][0]
                elif model_type == "clip_vision":
                    base_model = folder_paths.folder_names_and_paths["clip_vision"][0][0]
                elif model_type == "gligen":
                    base_model = folder_paths.folder_names_and_paths["gligen"][0][0]
                elif model_type == "upscale":
                    base_model = folder_paths.folder_names_and_paths["upscale_models"][0][0]
                elif model_type == "embeddings":
                    base_model = folder_paths.folder_names_and_paths["embeddings"][0][0]
                else:
                    base_model = "etc"
            return base_model

        row = df[df['url'] == url]
        if type(row) == pd.DataFrame:
            row = row.iloc[0]
        new_folder = Path(get_model_dir(row))
        new_dest = new_folder / row['filename']
        return new_folder, new_dest

    sema = asyncio.BoundedSemaphore(processes)
    dl_try_again = {}
    df_comfy_ui_mngr = load_df_cm(bp)

    async def fetch_file(session, url_item):
        file_url, file_dir = url_item
        file_dir = Path(file_dir)
        file_dir.mkdir(exist_ok=True)
        r_dict = {}

        async with sema:
            async with session.get(file_url) as resp:
                if file_url in df_comfy_ui_mngr['url'].unique():
                    file_dir, file_name = replace_dest(df_comfy_ui_mngr, file_url)
                    file_dir.mkdir(exist_ok=True)
                else:
                    file_name = file_url.split("/")[-1]
                    if 'Content-Disposition' in list(resp.headers):
                        regex_search = re.search(
                            'filename\=[\\\'|"](.*)[\\\'|"]', resp.headers['Content-Disposition']) # type: ignore
                        if regex_search:
                            file_name = regex_search.group(1)
                    file_name = file_dir / file_name

                if not resp.status == 200:
                    # print(
                    #     f'\nCould not download from URL: {file_url} (Status:{resp.status}) '
                    #     'Updating dl_try_again.'
                    # )
                    if Path(file_name).exists():
                        os.remove(Path(file_name).absolute().as_posix())
                        r_dict = {file_url: file_dir}
                    return 0, r_dict
        
                if not file_name.exists():
                    size = int(resp.headers.get('content-length', 0)) or 0
                    try:
                        async with aiofile.async_open(file_name.absolute().as_posix(), mode="wb") as outfile:
                            async for chunk in resp.content.iter_chunked(1024):                    
                                await outfile.write(chunk)
                    except Exception as error: # asyncio.exceptions.TimeoutError
                        # print('-'*25)
                        # print(f'Error on file: {file_name}.\n(URL: {file_url}).\nRerun file after completion.')
                        # print('Error:', error)
                        # print(f'Updating dl_try_again. (File name: {file_name}). (File URL: {file_url})')
                        r_dict = {file_url: file_dir}

                        # print(resp.headers)
                        # print('-'*25)
                        if Path(file_name).exists():
                            os.remove(Path(file_name).absolute().as_posix())
                        size = 0
                else:
                    print(f"\n{file_name.name} exists, skipping download.")
                    size = 0
        return size, r_dict

    async def download_files():
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_read=None, sock_connect=None, connect=None)) as session:
            tasks = [fetch_file(session, url_item) for url_item in url_dict.items()]
            return await tqdm.asyncio.tqdm.gather(*tasks, desc='Models Downloaded')

    start_time = time.time()
    loop = asyncio.get_event_loop()
    list_to_upack = loop.run_until_complete(download_files())
    download_size = [a for a, b in list_to_upack]
    dl_try_again  = [b for a, b in list_to_upack]

    dl_try_again_dict = {}
    for item in dl_try_again:
        if item != {}:
            try:
                dl_try_again_dict.update(item)
            except:
                print(f'err: {item}')
                raise ValueError(item)

    time_elapsed = time.time() - start_time
    return time_elapsed, sum(download_size), dl_try_again_dict

def print_stats(time_elapsed, download_size):
    """
    Prints stats from the function download_files
    """
    
    out_str = 'Time elapsed: ' 
    if time_elapsed <= 60:
        out_str += f'{time_elapsed:.1f} seconds'
    else:
        out_str += f'{time_elapsed // 60:.0f} minutes and {time_elapsed % 60:.0f} seconds'

    out_str += '\n' + 'Download size: '
    if download_size >= 1e9:
        out_str += f'{download_size * 1e-9:.2f} GB'
    else:
        out_str += f'{download_size * 1e-6:.2f} MB'

    out_str += '\n' + f'Download Speed: {download_size * 1e-6 / time_elapsed:.2f} MB/s'
    print(out_str)

def format_model_downloads(hf_models, civit_models, misc_link_models):
    url_dict = {}
    for key, items in hf_models.items():
        for inner_item in items:
            for idx in range(1, len(inner_item)):
                op = bp/f'comfy-ui/models/{key}' if type(key) == str else key
                url_dict.update({f'https://huggingface.co/{inner_item[0]}/resolve/main/{inner_item[idx]}': op})
    for key, items in civit_models.items():
        for item in items:
            op = bp/f'comfy-ui/models/{key}' if type(key) == str else key
            url_dict.update({f'https://civitai.com/api/download/models/{item}': op})
    for key, items in misc_link_models.items():
        for item in items:
            op = bp/f'comfy-ui/models/{key}' if type(key) == str else key
            url_dict.update({item: op})
    return url_dict

if __name__ == '__main__':

    os.chmod(Path(bp) / 'installer_files', 0o777) # Gives permissions for anyone to delete the installer file folder

    if not Path(bp / 'comfy-ui').exists():
        git_clone("comfyanonymous/ComfyUI.git" , f"{bp}/comfy-ui")

    for dir in ['inputs-comfy-ui', 'outputs-comfy-ui', 'workflows', 'comfy-ui/models/insightface']:
        (bp / dir).mkdir(exist_ok=True)
    
    # Setting up a Linux distribution

    if platform.system() == 'Linux':
        cmds = [
            "apt-get update",
            "apt-get install libglfw3-dev libgles2-mesa-dev freeglut3-dev " +
            "build-essential ffmpeg libssl-dev libffi-dev python3-dev " +
            "libsndfile1 p7zip nano",
            "apt-get upgrade -y"
        ]
        try:
            cc(' && '.join(cmds))
        except:
            cc(' && '.join([f'sudo {cmd}' for cmd in cmds]))

        # Cloudflared allows users to create a public link for a ComfyUI instance.
        # Executing the file share_port_linux.sh on Linux will allow users to share port 3000
        # with a randomly generated public link.

        try:
            _=check_call(('cloudflared', '--help'), stdout=PIPE) # Check to see if package is installed
        except:
            cc('wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb')
            try:
                cc('dpkg -i cloudflared-linux-amd64.deb')
            except:
                cc('sudo dpkg -i cloudflared-linux-amd64.deb')
            cc('rm cloudflared-linux-amd64.deb')

    # Writing files to open and share ComfyUI for Linux and Windows

    sh_file_p = bp / 'run_comfy_ui_linux.sh'
    if not sh_file_p.exists():
        with open(sh_file_p, 'w') as text_file:
            text_file.write(
                "MESA_GL_VERSION_OVERRIDE=4.1\n" # Fixes an issue on WSL2 for AnimateDiff
                f"cd {(bp / 'comfy-ui').absolute().as_posix()}\n"
                f"{venv_path.absolute().as_posix()} main.py --port 3000 "
                f"--highvram --listen --input-directory inputs-comfy-ui "
                "--output-directory outputs-comfy-ui"
            )
        st = os.stat(sh_file_p)
        os.chmod(sh_file_p, st.st_mode | stat.S_IEXEC)

    sh_file_c = bp / 'share_port_linux.sh'
    if not sh_file_c.exists():
        with open(sh_file_c, 'w') as text_file:
            text_file.write(f'cloudflared tunnel --url http://127.0.0.1:3000')
        st = os.stat(sh_file_c)
        os.chmod(sh_file_p, st.st_mode | stat.S_IEXEC)
    
    bat_file_p = bp / 'run_comfy_ui_windows.bat'
    if not bat_file_p.exists():
        with open(bat_file_p, 'w') as text_file:
            text_file.write(
                f"cd %0\.."
                f"{sys.executable} " 
                f"{(bp / 'comfy-ui/main.py').absolute().as_posix()} " # "%0\..\comfy-ui\main.py "
                f"--input-directory  {(bp / 'inputs-comfy-ui' ).absolute().as_posix()} "
                f"--output-directory {(bp / 'outputs-comfy-ui').absolute().as_posix()} "
                "--windows-standalone-build"
            )
            st = os.stat(bat_file_p)
            os.chmod(bat_file_p, st.st_mode | stat.S_IEXEC)

    print(sys.executable) 

    pip_install('--upgrade', 'pip')
    pip_install("torch", "torchvision", "torchaudio",
                "--extra-index-url=https://download.pytorch.org/whl/cu118")
    pip_install("-r", f"{bp.absolute().as_posix()}/comfy-ui/requirements.txt")
    pip_install("boltons")
    pip_install("asyncio", "aiohttp", "aiofile",  "tqdm", "requests", "pandas")
    import asyncio, aiofile, aiohttp, tqdm, tqdm.asyncio, requests, pandas as pd
    if 'melMass/comfy_mtb' in extensions:
        pip_install("tensorflow", "facexlib", "insightface", "basicsr")
    # Installing ComfyUI Extensions
    
    url_dict = format_model_downloads(hf_models, civit_models, misc_link_models)

    mtb_path = Path(bp.absolute().as_posix() + '/comfy-ui/custom_nodes/comfy_mtb/')
    mtb_path_exists_before = mtb_path.exists()

    for repositories in extensions:
        if type(repositories) == str:
            repositories = [repositories]
        else:
            repositories = [f"{repositories[0]}/{r}" for r in repositories[1:]]
        for repository in repositories:
            repository_loc = Path(f"{bp}/comfy-ui/custom_nodes/{repository.split('/')[-1]}")
            if not repository_loc.exists():
                git_clone(repository, repository_loc)
            if repository.split('/')[-1] in repositories_without_lazy_install:
                requirements_location = repository_loc / "requirements.txt"
                if requirements_location.exists():
                    pip_install("-r", requirements_location)

    # comfy_mtb needs to run an extra script to use all nodes
    
    if 'melMass/comfy_mtb' in extensions and mtb_path_exists_before == False and mtb_path.exists():
        try:
            cc(f'{sys.executable} '
               f'{str((bp/"comfy-ui/custom_nodes/comfy_mtb/scripts/download_models.py").absolute().as_posix())} -y')
        except:
            comfy_mtb_path = str(Path(f"{bp}/comfy-ui/custom_nodes/{repository.split('/')[-1]}").absolute().as_posix())
            os.chmod(comfy_mtb_path, 0o777)
            rmtree(comfy_mtb_path)
            print('Failed to install extension `melMass/comfy_mtb`, please use the ComfyUI Manager Menu inside '
                  'the ComfyUI program to install this extension.')
            
    sys.path.insert(1, str(Path(bp / 'comfy-ui').absolute().as_posix()))
    import folder_paths
    sys.path.remove(str(Path(bp / 'comfy-ui').absolute().as_posix()))

    # Downloading ComfyUI Models

    time_elapsed, download_size, dl_try_again = download_files(url_dict, processes=5)
    print('Download #1')
    print_stats(time_elapsed, download_size)

    if len(dl_try_again) != 0:
        time_elapsed, download_size, dl_try_again = download_files(dl_try_again, processes=2)
        print(f'Download #2 {len(dl_try_again)}')
        print_stats(time_elapsed, download_size)

        if len(dl_try_again) != 0:
            time_elapsed, download_size, dl_try_again = download_files(dl_try_again, processes=1)
            print(f'Download #3 {len(dl_try_again)}')
            print_stats(time_elapsed, download_size)

            if len(dl_try_again) != 0:
                dl_try_again_str = '\n'.join([
                    key # f"{item.parents[1].name}/{item.parent.name}/{item.name}: {key}"
                    for key, item in dl_try_again.items()
                ])
                print(f'Could not download urls\n{dl_try_again_str}\n\n in three attempts.')

    if len(dl_try_again) == 0:
        print('Downloaded all files')
