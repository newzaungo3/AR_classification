LABEL ={1:'SCGC', 2:'SCGP',3:'WEDO',
        4:'PRINGLES',5:'HONDA',6:'LOUIS VUITTON',7:'APPMAN',
        8:'RIOT',9:'NIKE',10:'BMW',11:'FORD',12:'CAFE AMAZON',
        13:'MCDONALD',14:'KFC',15:'STARBUCKS',16:'BURGER KING',
        17:'DELL',18:'COCA COLA',19:'ADIDAS',20:'PEPSI',
        21:'A&W',22:'LAZADA',23:'INTEL',24:'SHOPEE',25:'ALLWELL',
        26:'CARRIER',27:'DAIKIN',28:'CORSAIR',29:'KINGSTON',30:'SAVEPAK',
        31:'SENNHEISER',32:'SEAGATE',33:'ASICS',34:'MALEE',35:'CALVIN KLEIN',36:'CAMPER',
        37:'CHAMPION',38:'DIOR',39:'CROCS',40:'H&M',41:'FILA',42:'CONVERSE',43:'GUESS',
        44:'LACOSTE',45:'NEW BALANCE',46:'UNIQLO',47:'COINGECKO',48:'CMC',49:'KRISPY KREME',50:'PUMA',
        51:'LAYS',52:'M&M',53:'MAGGI',54:'MAGNUM',55:'KITKAT',56:'HERSHEY',57:'HAAGEN-DAZS',58:'HEINZ',59:'CHUPA CHUPS',60:'DORITOS',
        61:'ORERO',62:'NUTELLA',63:'SHOPEE FOOD',64:'SNICKERS',65:'TOBLERONE',66:'3M',67:'COLGATE',68:'COOLER MASTER',69:'DENSO',70:'ESSO',
        71:'EXXON',72:'IKEA',73:'MICHELIN',74:'MOBIL',75:'PANASONIC',76:'PFIZER',77:'TEFAL',78:'AMAZON',79:'PRIME',80:'AIRBNB',
        81:'CLICKUP',82:'CLOUD 9',83:'EBAY',84:'DROPBOX',85:'FACEBOOK',86:'GOOGLE',87:'IMDB',88:'KAHOOT',89:'LINKEDIN',90:'MIRO',
        91:'PINTEREST',92:'PATREON',93:'QUILLBOT',94:'SHUTTERSTOCK',95:'SLACK',96:'UBER',97:'VIMER',98:'YOUTUBE',
        99:'ZOOM',100:'EA',101:'Undefined'}


CLIP_BACKBONE = 'RN50'
CLIP_ONNX_EXPORT_PATH = 'clip_coca.onnx'
CLIP_ONNX_EXPORT_PATH_SIMP = 'clip_coca_simplified.onnx'

ONNX_INPUT_NAMES = ["IMAGE"]
ONNX_OUTPUT_NAMES = ["LOGITS_PER_IMAGE"]
ONNX_DYNAMIC_AXES = {
    "IMAGE": {
        0: "image_batch_size",
    },
    "LOGITS_PER_IMAGE": {
        0: "image_batch_size",
        1: "text_batch_size",
    }
}