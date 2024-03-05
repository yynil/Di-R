import torch
from PIL import Image
import open_clip
import pandas as pd

def extract_annotations_file(input_annotation_file, output_pickle_file,tokenizer,  batch_size=256):
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
    input_ids = []
    with torch.no_grad():
        progress_bar = tqdm(range(0, len(annotations), batch_size),desc=f'Processing {input_annotation_file}')
        for i in progress_bar:
            progress_bar.set_description(f"Processing {i} - {i + batch_size}")
            batch = annotations[i:i + batch_size]
            # print(batch)
            texts = [x['caption'] for x in batch]
            # print(texts_features.shape)
            image_ids = [x['image_id'] for x in batch]
            ids = [x['id'] for x in batch]
            image_id.extend(image_ids)
            id.extend(ids)
            features.extend(texts)
            tokenized_ids = [tokenizer.encode(t) for t in texts]
            input_ids.extend(tokenized_ids)
    print(len(image_id), len(id), len(features),len(features[0]),len(input_ids))

    # Create a DataFrame with image_id, id, and features
    df = pd.DataFrame({'image_id': image_id, 'id': id, 'features': features,'input_ids':input_ids})
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
    input_ids = df.loc[image_id, 'input_ids']
    print(input_ids)

if __name__ == '__main__':
    import os
    from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
    current_path = os.path.dirname(__file__)
    dict_path = os.path.join(current_path, 'tokenizer', 'rwkv_vocab_v20230424.txt')
    tokenizer = TRIE_TOKENIZER(dict_path)
    print(tokenizer.encode('hello world'))
    input_annotation_file = '/media/yueyulin/KINGSTON/data/images/coco/captions_train2017.json'
    output_pickle_file = '/media/yueyulin/KINGSTON/data/images/coco/coco_captions_train2017_texts.pkl'
    extract_annotations_file(input_annotation_file, output_pickle_file,tokenizer)

    input_annotation_file = '/media/yueyulin/KINGSTON/data/images/coco/captions_val2017.json'
    output_pickle_file = '/media/yueyulin/KINGSTON/data/images/coco/captions_val2017_texts.pkl'
    extract_annotations_file(input_annotation_file, output_pickle_file,tokenizer)


