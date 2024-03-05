import torch
from PIL import Image
import open_clip
import pandas as pd

def extract_annotations_file(input_annotation_file, output_pickle_file,model, tokenizer, batch_size=256):
    import json
    with open(input_annotation_file, 'r') as f:
        data = json.load(f)
    annotations = data['annotations']
    from tqdm import tqdm
    print(len(annotations))
    print(annotations[0])
    image_id = annotations[0]['image_id']
    image_id = []
    id = []
    features = []
    with torch.no_grad():
        progress_bar = tqdm(range(0, len(annotations), batch_size),desc=f'Processing {input_annotation_file}')
        for i in progress_bar:
            progress_bar.set_description(f"Processing {i} - {i + batch_size}")
            batch = annotations[i:i + batch_size]
            # print(batch)
            texts = [x['caption'] for x in batch]
            texts = tokenizer(texts).to('cuda')
            texts_features = model.encode_text(texts)
            # print(texts_features.shape)
            image_ids = [x['image_id'] for x in batch]
            ids = [x['id'] for x in batch]
            image_id.extend(image_ids)
            id.extend(ids)
            features.extend(texts_features.tolist())
            # break
            # images = [Image.open(x['file_name']) for x in batch]
            # images = [preprocess(x) for x in images]
            # images = torch.stack(images).to('cuda')
            # print(images.shape)
            # print(images)
            # break
    print(len(image_id), len(id), len(features),len(features[0]))

    # Create a DataFrame with image_id, id, and features
    df = pd.DataFrame({'image_id': image_id, 'id': id, 'features': features})
    df.set_index('image_id', inplace=True)
    # Save the DataFrame to disk in an efficient binary format
    df.to_pickle(output_pickle_file)

    # Load the DataFrame from disk
    df = pd.read_pickle(output_pickle_file)
    print(df)
    
    # build a index using image_id
    # df.set_index('image_id', inplace=True)

    # Get the features for a specific image

    features = df.loc[image_id, 'features']
    print(features)

if __name__ == '__main__':
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k',device='cuda')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    print(model, preprocess, tokenizer)
    input_annotation_file = '/media/yueyulin/KINGSTON/data/images/coco/captions_train2017.json'
    output_pickle_file = '/media/yueyulin/KINGSTON/data/images/coco/coco_captions_train2017.pkl'
    extract_annotations_file(input_annotation_file, output_pickle_file, model, tokenizer)

    input_annotation_file = '/media/yueyulin/KINGSTON/data/images/coco/captions_val2017.json'
    output_pickle_file = '/media/yueyulin/KINGSTON/data/images/coco/captions_val2017.pkl'
    extract_annotations_file(input_annotation_file, output_pickle_file, model, tokenizer)


