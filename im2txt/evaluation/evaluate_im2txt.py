from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

# set up file names and pathes
dataDir='../data/mscoco/raw-data/'
dataType='val2014'
algName = 'im2txt'
annFile='%s/annotations/captions_%s.json'%(dataDir,dataType)
resFile= './results/captions_%s_%s_results.json'%(dataType, algName)

# create coco object and cocoRes object
coco = COCO(annFile)
cocoRes = coco.loadRes(resFile)

# create cocoEval object by taking coco and cocoRes
cocoEval = COCOEvalCap(coco, cocoRes)

# evaluate on a subset of images by setting
# cocoEval.params['image_id'] = cocoRes.getImgIds()
# please remove this line when evaluating the full validation set
cocoEval.params['image_id'] = cocoRes.getImgIds()

# evaluate results
cocoEval.evaluate()

