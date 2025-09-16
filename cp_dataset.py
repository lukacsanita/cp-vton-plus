# coding=utf-8
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from PIL import ImageDraw

import os.path as osp
import numpy as np
import json


class CPDataset(data.Dataset):
    """Dataset for CP-VTON+.

    This class loads and preprocesses data for the CP-VTON+ model.
    It supports different stages (GMM and TOM) and datamodes (train, test).
    """

    def __init__(self, opt):
        super(CPDataset, self).__init__()
        # Base configuration
        self.opt = opt
        self.root = opt.dataroot
        self.datamode = opt.datamode  # train or test or self-defined
        self.stage = opt.stage  # GMM or TOM
        self.data_list = opt.data_list
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.radius = opt.radius
        self.data_path = osp.join(opt.dataroot, opt.datamode)
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(), # numpy/PIL → tensor, és [0, 255] → [0, 1],
            transforms.Normalize((0.5,), (0.5,))]) # [0, 1] → [-1, 1].

        # Load image - cloth pairs from a text file.
        im_names = []
        c_names = []
        with open(osp.join(opt.dataroot, opt.data_list), 'r') as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)

        self.im_names = im_names
        self.c_names = c_names

    def name(self):
        return "CPDataset"

    def __len__(self):
        return len(self.im_names)

    def __getitem__(self, index):
        c_name = self.c_names[index] #aktuális ruha neve
        im_name = self.im_names[index] #aktuális személy képfájljának neve
        if self.stage == 'GMM':
            c = Image.open(osp.join(self.data_path, 'cloth', c_name))
            cm = Image.open(osp.join(self.data_path, 'cloth-mask', c_name)).convert('L') #grayscale
        else:
            c = Image.open(osp.join(self.data_path, 'warp-cloth', im_name))    # c_name, if that is used when saved
            cm = Image.open(osp.join(self.data_path, 'warp-mask', im_name)).convert('L')    # c_name, if that is used when saved

        c = self.transform(c)  # [-1, 1] # A ruha képét átalakítjuk egy normalizált Tensor-rá, amely a modell bemenete lehet.
        cm_array = np.array(cm)                         # A maszkot NumPy tömbbé konvertálja, 
        cm_array = (cm_array >= 128).astype(np.float32) # binarizálja (128 küszöb)
        cm = torch.from_numpy(cm_array)                  # [0, 1] # Majd visszaalakítja PyTorch tensorrá, 
        cm.unsqueeze_(0)                                 # és dimenziót bővít ([1, Height, Width] alakra). 0 --> sort ad

        # Load person image and apply transformation
        im = Image.open(osp.join(self.data_path, 'image', im_name))
        im = self.transform(im)  # [-1, 1] #Átalakítja ugyanolyan transzformációval

        """
        LIP labels
        
        [(0, 0, 0),    # 0=Background
         (128, 0, 0),  # 1=Hat
         (255, 0, 0),  # 2=Hair
         (0, 85, 0),   # 3=Glove
         (170, 0, 51),  # 4=SunGlasses
         (255, 85, 0),  # 5=UpperClothes
         (0, 0, 85),     # 6=Dress
         (0, 119, 221),  # 7=Coat
         (85, 85, 0),    # 8=Socks
         (0, 85, 85),    # 9=Pants
         (85, 51, 0),    # 10=Jumpsuits
         (52, 86, 128),  # 11=Scarf
         (0, 128, 0),    # 12=Skirt
         (0, 0, 255),    # 13=Face
         (51, 170, 221),  # 14=LeftArm
         (0, 255, 255),   # 15=RightArm
         (85, 255, 170),  # 16=LeftLeg
         (170, 255, 85),  # 17=RightLeg
         (255, 255, 0),   # 18=LeftShoe
         (255, 170, 0),   # 19=RightShoe
         (170, 170, 50)   # 20=Skin/Neck/Chest (Newly added after running dataset_neck_skin_correction.py)
        ]
         
        """

        # load parsing image
        parse_name = im_name.replace('.jpg', '.png')
        im_parse = Image.open(
            # osp.join(self.data_path, 'image-parse', parse_name)).convert('L')
            osp.join(self.data_path, 'image-parse-new', parse_name)).convert('L')   # updated new segmentation, Betölti a szegmentációs képet (image-parse-new), ahol minden pixel egy testrészhez tartozó címkét hordoz (pl. fej, kar, felsőruha).
        parse_array = np.array(im_parse) # Betölti a testmaszkot (fekete-fehér bináris kép), ami a CP-VTON+ esetében meghatározza az ember alakját.
        im_mask = Image.open(osp.join(self.data_path, 'image-mask', parse_name)).convert('L')
        mask_array = np.array(im_mask)

        # parse_shape = (parse_array > 0).astype(np.float32)  # CP-VTON body shape
        # Get shape from body mask (CP-VTON+)
        parse_shape = (mask_array > 0).astype(np.float32) # A parse_shape a test kontúr bináris maszkja ([0,1] értékekkel).

        #A parse_head egy bináris maszk a fejre és bizonyos testrészekre (GMM-nél kevesebb, TOM-nál több régió).
        if self.stage == 'GMM':
            parse_head = (parse_array == 1).astype(np.float32) + (parse_array == 4).astype(np.float32) + (parse_array == 13).astype(np.float32)  # CP-VTON+ GMM input (reserved regions)
        else:
            parse_head = (parse_array == 1).astype(np.float32) + \
                (parse_array == 2).astype(np.float32) + \
                (parse_array == 4).astype(np.float32) + \
                (parse_array == 9).astype(np.float32) + \
                (parse_array == 12).astype(np.float32) + \
                (parse_array == 13).astype(np.float32) + \
                (parse_array == 16).astype(np.float32) + \
                (parse_array == 17).astype(
                np.float32)  # CP-VTON+ TOM input (reserved regions)

        parse_cloth = (parse_array == 5).astype(np.float32) + (parse_array == 6).astype(np.float32) + (parse_array == 7).astype(np.float32)    # upper-clothes labels


        # shape downsample ---------------------------------------------------------------------------------
        # A parse_shape két lépéses le- és felskálázással simítja a maszkot. A shape, shape_ori, phead, pcm mind tensorrá alakul.
            #parse_shape itt egy bináris maszk float32-ként 0 és 1 értékekkel, amely a test sziluettjét (vagy háttér nélküli test alakját) ábrázolja.
            #egy normál (szürkeárnyalatos) képet csinál a test alakjának maszkjából.
        parse_shape_ori = Image.fromarray((parse_shape*255).astype(np.uint8)) # felszorozza 255-tel → [0, 255] skálára teszi (képformátumhoz), --> astype(np.uint8) → 8 bites képformátumra konvertálja, --> Image.fromarray(...) → PIL képpé alakítja a numpy tömböt.
            
            #A testmaszk képet lekicsinyíti 16-szorosára, bilineáris interpolációval (simább átskálázás).
        parse_shape = parse_shape_ori.resize(
            (self.fine_width//16, self.fine_height//16), Image.BILINEAR)

            #A lekicsinyített képet újra visszanagyítja az eredeti méretre, Ez a trükk elmosódást hoz létre → "puhább", simított testforma képet eredményez → ez hasznos lehet a modellnek, hogy általános alakstruktúrát tanuljon a konkrét részletek helyett.
        parse_shape = parse_shape.resize(
            (self.fine_width, self.fine_height), Image.BILINEAR)

            #Az eredeti, 255-ös skálán lévő bináris képet is átskálázza az elvárt fine_width és fine_height méretre. Ez a kép élesebb kontúrokat tartalmaz, nem volt kicsinyítve.
        parse_shape_ori = parse_shape_ori.resize(
            (self.fine_width, self.fine_height), Image.BILINEAR)
            
            #Mindkét képet átviszi egy transform lépésen, amit a konstruktorban a transforms.Compose([...]) definiál. Ez normálisan a következőket tartalmazza: - ToTensor() → numpy/PIL → tensor, és [0, 255] → [0, 1],  - Normalize((0.5,), (0.5,)) → [0, 1] → [-1, 1].
        shape_ori = self.transform(parse_shape_ori)  # [-1,1]
        shape = self.transform(parse_shape)  # [-1,1]
        
        phead = torch.from_numpy(parse_head)  # [0,1] -- egyszerűen torch.Tensor-rá alakítja.
        # phand = torch.from_numpy(parse_hand)  # [0,1]
        pcm = torch.from_numpy(parse_cloth)  # [0,1] -- A ruha maszkolt részét (egy bináris maszk [1, H, W], ami 1 ott, ahol ruha van, és 0 máshol) is átkonvertálja Tensorrá.

        #-------------------------------------------------------------------------------------------------------
        # Upper cloth -- Kivágja a ruhát az eredeti képből
        im_c = im * pcm + (1 - pcm)  # [-1,1], fill 1 for other parts, ruharégió + "fehér" háttér:
            #im * pcm: megtartja csak a ruhát az eredeti képből (mert ott, ahol pcm == 1, az im pixele megmarad, máshol lenullázódik)
            #(1 - pcm): ott 1, ahol nem ruha → ha ezt hozzáadjuk, akkor ezek a pixelek fehér vagy 1 értéket kapnak (attól függően, mi a formátum)
            #Tehát a végeredmény: Ahol ruha van: megtartja az eredeti színeket az im-ből, Ahol nincs ruha: a háttér világos vagy fehér lesz (függően a normalizálástól)

        # fejet is kivágja
        im_h = im * phead - (1 - phead)  # [-1,1], fill -1 for other parts, fej régió + "-1" háttér (fekete)

        
        # load pose points ---------------------------------------------------------------------------
        pose_name = im_name.replace('.jpg', '_keypoints.json') # Az aktuális képfájl nevéből (im_name, pl. 00001_00.jpg) a hozzá tartozó OpenPose keypoint fájl nevét generálja: 00001_00_keypoints.json.
        
        with open(osp.join(self.data_path, 'pose', pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints'] # Az OpenPose mindig 'people' tömböt ad vissza (még akkor is, ha csak 1 ember van), így az első embert ([0]) használja. pose_keypoints egy hosszú lista: minden testrész 3 értékkel szerepel benne: x, y, confidence.
            pose_data = np.array(pose_data) # Átalakítja a JSON-ból kiolvasott listát egy (N, 3) alakú NumPy tömbbé. N = 18 keypoint (vagy több, ha hands/face is lenne), oszlopok: x, y, conf.
            pose_data = pose_data.reshape((-1, 3)) # ezt átalakítod egy (N, 3) méretű tömbbé a reshape-pel --> shape[0] → a kulcspontok száma.
            #Az OpenPose által visszaadott pose_keypoints lista minden egyes kulcspontot (pl. fej, nyak, váll, térd) egy 3 értékből álló egységként ír le: [x, y, confidence] → tehát: a kulcspont helye és a pontossága.
            #Ez a lista 3 * N hosszú, ahol N a kulcspontok száma.

        #Létrehoz egy pose_map nevű tenzort, amelyben minden testrészhez (pl. fej, váll, térd) lesz egy külön "csatorna" (mint egy fekete-fehér kép). Méret: [point_num, H, W] (pl. [18, 256, 192]), értéke alapból 0.
        point_num = pose_data.shape[0] #testkulcspontok (keypoints) száma
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
        
        r = self.radius
        im_pose = Image.new('L', (self.fine_width, self.fine_height)) # egy összesített fekete-fehér PIL kép, ahova majd minden pontot kirajzol.
        pose_draw = ImageDraw.Draw(im_pose) #ehhez hoz létre rajzoló objektumot.
            #Végigmegy minden kulcsponton (keypoint): pl. nyak, váll, könyök, térd, boka... Létrehoz egy új fekete-fehér képet minden ponthoz: one_map.
        for i in range(point_num):
            one_map = Image.new('L', (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)
            pointx = pose_data[i, 0]
            pointy = pose_data[i, 1]
            if pointx > 1 and pointy > 1:  # Ha a keypoint értelmes (nem 0 vagy 1, tehát nem hiányzik), akkor egy fehér négyzetet rajzol a képen a pont körül
                #Két helyre is rajzol: -- one_map → ez egyetlen kulcspont maszkja, --im_pose → ez az összes kulcspont közös képe (diagnosztikához, megjelenítéshez).
                draw.rectangle((pointx-r, pointy-r, pointx +
                                r, pointy+r), 'white', 'white')
                pose_draw.rectangle(
                    (pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
                
            one_map = self.transform(one_map) #A one_map képet (csak 1 kulcspont maszkja) átkonvertálja Tensorrá a szokásos transformmal
            pose_map[i] = one_map[0] #A pose_map[i] helyére beteszi.
            #Minden egyes pózpont egy csatornát kap, ahol csak ott van érték, ahol a pont megjelent.
            
        #Ez a rész létrehozza a pose_map tenzort: Mérete: [point_num, H, W] → minden kulcspont (pl. nyak, térd, stb.) külön csatornában. Ezek a csatornák fehér négyzeteket tartalmaznak a kulcspont helyén.
        #Ez segíti a modellt, hogy a testtartás alapján döntse el, hová illeszkedjen a ruha (pl. ha kezet felemeli a személy, a ruha ujja is másképp álljon).
        #--------------------------------------------------------------------
        
        # Just for visualization, nem megy be a modellbe
        im_pose = self.transform(im_pose)

        # cloth-agnostic representation---------------------------------------
        # A ruha-agnosztikus reprezentáció (clothing-agnostic representation) 22 (vagy több) csatornás tensor: -shape (éles emberi forma): 1 csatorna, -im_h (fej): 3 csatorna (RGB fej), -pose_map: 18 csatorna (valszeg több ld. LIP komment)
        agnostic = torch.cat([shape, im_h, pose_map], 0) # Ezeket torch.cat([...], 0) összeilleszti a csatorna dimenzióban → [20, H, W] alakú tensor. 

        if self.stage == 'GMM': # Ha épp a GMM (ruha-igazítás) fázisban vagyunk: Beolvas egy rácskép-et (grid.png) – ez egy geometriai referencia, amit vizualizációra és háborítások figyelésére használnak. 
            im_g = Image.open('grid.png')
            im_g = self.transform(im_g) # Átalakítja tenzorrá.
        else:
            im_g = ''

        pcm.unsqueeze_(0)  # CP-VTON+, parse_cloth_mask, vagyis a képen lévő eredeti ruha maszkja (bináris). .unsqueeze_(0) hozzáad egy batch dimenziót → alak: [1, H, W],  Ez hasznos lehet, ha a modellt batch-es feldolgozásra készítjük fel.

        result = {
            'c_name':   c_name,     # for visualization, a ruhadarab fájlneve (str)
            'im_name':  im_name,    # for visualization or ground truth, a személy (modell) képfájlneve - str
            'cloth':    c,          # for input, az input ruhadarab RGB képe (Tensor [3, H, W])
            'cloth_mask':     cm,   # for input(GMM), előfeldolgozáshoz,  bináris maszk a ruhához (hol van ruha, hol nem) (Tensor [1, H, W])
            'image':    im,         # for visualization vagy ha supervision kell, az eredeti személy képe (egész test, ruhában) (Tensor [3, H, W])
            'agnostic': agnostic,   # for input, 1 csatorna: shape – sziluett, 3 csatorna: head – fej RGB-ben, 18 csatorna: pose_map – testkulcspontok
            'parse_cloth': im_c,    # for ground truth, az eredeti képen lévő ruha területének kivágott RGB képe (Tensor [3, H, W])
            'shape': shape,         # for visualization, a személy sziluettje (bináris maszk) (Tensor [1, H, W])
            'head': im_h,           # for visualization, a fej RGB kivágása az eredeti képből (Tensor [3, H, W])
            'pose_image': im_pose,  # for visualization, kulcspontokból generált fekete-fehér kép (Tensor [1, H, W])
            'grid_image': im_g,     # for visualization, rácskép, ha a GMM szakaszban vagyunk
            'parse_cloth_mask': pcm,     # for CP-VTON+, TOM input,  személy eredeti ruhájának bináris maszkja
            'shape_ori': shape_ori,     # original body shape without resize,  az eredeti testforma (sziluett), de nem downsample-ölve, tehát teljes felbontásban
        }

        return result


class CPDataLoader(object):
    """
    This class creates a data loader for training/testing the CP-VTON+ model.
    """

    def __init__(self, opt, dataset):
        super(CPDataLoader, self).__init__()

        # Define sampler based on shuffle option
        train_sampler = None
        if opt.shuffle:
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)

        # A DataLoader a híd a nyers adatok (Dataset) és a tanítási ciklus között. Segít a batch-elésben, shuffl-olásban és hatékony adatbetöltésben.
        # torch.utils.data.DataLoader a PyTorch egyik leggyakrabban használt osztálya, amely a dataset-ek (adathalmazok) kezelését és betöltését könnyíti meg gépi tanulási modellekhez.
        self.data_loader = torch.utils.data.DataLoader( # Meghívja a CPDataset.__getitem__(index)-et minden egyes elemre,
            dataset, 
            batch_size=opt.batch_size, # Ha batch_size=4, akkor 4-szer hívja meg a __getitem__() metódust,
            shuffle=(train_sampler is None),
            num_workers=opt.workers, # mennyi paralel adatbetöltő szálat (num_workers) használjon
            pin_memory=True, # legyen-e GPU-memóriás gyorsítás
            sampler=train_sampler
        )
        self.dataset = dataset

        # Initialize the data iterator
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            # Reset the iterator if reaching the end of the dataset
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch


if __name__ == "__main__":
    print("Check the dataset for geometric matching module!")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="data", help="Path to the directory containing the dataset")
    parser.add_argument("--datamode", default="train", help="Data processing mode (train or test)")
    parser.add_argument("--stage", default="GMM")
    parser.add_argument("--data_list", default="train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--shuffle", action='store_true', help='Shuffle input data')
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    parser.add_argument('-j', '--workers', type=int, default=1, help='Number of worker threads for data loading')

    opt = parser.parse_args()
    dataset = CPDataset(opt)
    data_loader = CPDataLoader(opt, dataset)

    print('Size of the dataset: %05d, dataloader: %04d'
          % (len(dataset), len(data_loader.data_loader)))
    first_item = dataset.__getitem__(0)
    first_batch = data_loader.next_batch()

    from IPython import embed
    embed()
