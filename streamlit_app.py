import streamlit as st
import cv2
import os
import moviepy.video.io.ImageSequenceClip
from natsort import os_sorted
from hsemotion.facial_emotions import HSEmotionRecognizer
import shutil
import pathlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch  # it is important to import torch before insightface because of onnx
import insightface
from insightface.app import FaceAnalysis
import mmcv
from PIL import Image, ImageDraw
from scenedetect import detect, ContentDetector
from deep_sort_realtime.deepsort_tracker import DeepSort
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.7)

id_to_emo = {0: 'Anger', 1: 'Contempt', 2: 'Disgust', 3: 'Fear', 4: 'Happiness', 5: 'Neutral', 6: 'Sadness',
             7: 'Surprise'}


def detect_faces(img):
    return app.get(img)


def remap_bboxes_insightface(bbs):
    """
    Remaps box to ([[x0, y0, w,h], confidence, detection_class) format
    """
    final_tuple = []
    for ind in range(len(bbs)):
        box = bbs[ind]['bbox']  # [x0, y0, x1, y1]
        confidence = bbs[ind]['det_score']
        x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
        box_list = [[x0, y0, x1 - x0, y1 - y0], confidence,
                    ind]
        final_tuple.append(box_list)
    return tuple(final_tuple)


def create_embedding(img):
    final_tuple = []
    for ind in range(len(img)):
        embedding = img[ind]['embedding']
        final_tuple.append(embedding)
    return final_tuple


# %%
def extract_frames(video_path):
    video = mmcv.VideoReader(video_path)

    frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in
              video]
    return frames, video.fps


# %%
def process_one_image(frame):
    img = np.array(frame)
    all_dets = detect_faces(img)
    embeddings = create_embedding(all_dets)
    return all_dets, embeddings


# %%
def construct_json(track):
    return {
        'track_id': track.track_id,
        'bounding_box': track.to_ltrb()  # left_x, top_y, right_x, bottom_y
    }


def compare_embs(old_embs, new_emb, rec_model, thresh, track_id):
    max_id = track_id
    max_score = 0

    for k in old_embs.keys():

        score = rec_model.compute_sim(old_embs.get(k), new_emb)
        if score > max_score:
            max_score = score
            max_id = k
    if max_score < thresh:
        return track_id
    else:
        return max_id


def clear_submit():
    st.session_state["submit"] = False


uploaded_file = st.file_uploader(
    "",
    type=["mp4"],
    on_change=clear_submit,
)

val_folder = Path('../VGAF/Val')
file_name = Path(uploaded_file.name)

button = st.button("Submit")

if button:

    frames, fps = extract_frames(str(val_folder / file_name))

    save_interval = 10

    face_directory = 'face_dir'
    if os.path.exists(face_directory):
        shutil.rmtree(face_directory)
        os.makedirs(face_directory)

    num_lookup_frames = 5

    confusion_matrix = np.zeros((1000, 1000))
    frames_threshold = 10

    scene_frames_list = []

    frames_tracked = []
    tracker = DeepSort(max_age=4, max_cosine_distance=0.9, max_iou_distance=0.4, nn_budget=50, n_init=3)

    rec_model = app.models['recognition']
    frames_to_compute_sim = []
    start_time = time.time()
    for ind, frame in enumerate(frames):

        one_image_tracks = []

        frame_draw = frame.copy()
        draw = ImageDraw.Draw(frame_draw)

        all_dets, embeddings = process_one_image(frame)
        bbs = remap_bboxes_insightface(all_dets)

        tracks = tracker.update_tracks(bbs,
                                       embeds=embeddings)  # bbs expected to be a list of detections, each in tuples of ( [x0, y0, w,h], confidence, detection_class )

        one_track_faces = defaultdict(dict)
        one_track_faces_save = defaultdict(dict)

        for track in tracks:

            if not track.is_confirmed():
                continue

            track_info = construct_json(track)
            track_id, ltrb = track_info['track_id'], track_info[
                'bounding_box']

            if (ind > num_lookup_frames) and (ind not in scene_frames_list) and len(frames_to_compute_sim) > (
                    ind - (num_lookup_frames - 1)):
                if frames_to_compute_sim[ind - (num_lookup_frames - 1)].get(
                        str(track_id)) is None:  # condition of new face

                    old_track_id = track_id

                    track_id = compare_embs(frames_to_compute_sim[ind - (num_lookup_frames - 1)], track.get_feature(),
                                            rec_model, 0.3, track_id)

                    if old_track_id != track_id:
                        confusion_matrix[int(track_id)][int(old_track_id)] += 1

                    if confusion_matrix[int(track_id)][int(old_track_id)] > frames_threshold:
                        track_id = old_track_id
                        track.__setattr__('track_id', track_id)

            elif (ind > num_lookup_frames) and (ind in scene_frames_list) and len(frames_to_compute_sim) > (
                    ind - (num_lookup_frames - 1)):
                num_lookup_frames = 5
                if frames_to_compute_sim[ind - (num_lookup_frames - 1)].get(str(track_id)) is None:
                    old_track_id = track_id
                    track_id = compare_embs(frames_to_compute_sim[ind - (num_lookup_frames - 1)], track.get_feature(),
                                            rec_model, 0.3, track_id)
                    if old_track_id != track_id:
                        confusion_matrix[int(track_id)][int(old_track_id)] += 1

                    if confusion_matrix[int(track_id)][int(old_track_id)] > frames_threshold:
                        track_id = old_track_id
                        track.__setattr__('track_id', track_id)

            one_track_faces[track_id] = track.get_feature()
            one_track_faces_save[track_id] = {'features': track.get_feature(), 'box': ltrb.tolist()}

            if len(ltrb) != 4:
                continue
            one_image_tracks.append(ltrb)
            new_ltrb = ltrb.tolist()
            #     --------------------------------------------------------------------- draw boxes --------------------------------------------------------------------- #
            draw.rectangle([(new_ltrb[2], new_ltrb[3]), (new_ltrb[0], new_ltrb[1]), ], outline=(255, 0, 0),
                           width=6)  # red are predicted
            text_to_write = f'id: {track_id}'
            draw.text((new_ltrb[2], new_ltrb[3]), text_to_write, fill=(255, 0, 0), width=6)
            if not all_dets: continue
            for box in all_dets:
                box = box['bbox']
                draw.rectangle(box.tolist(), outline=(0, 255, 0), width=6)  # green are predicted by detector

        if len(one_track_faces_save) == 0: continue

        if ind % save_interval == 0:
            path_to_frame_faces = f'{face_directory}/{file_name.stem}/frame_{ind}'
            os.makedirs(path_to_frame_faces)
            for face_dict_num in one_track_faces_save.keys():
                features = one_track_faces_save[face_dict_num]['features']
                box = one_track_faces_save[face_dict_num]['box']

                np.save(f'{path_to_frame_faces}/{face_dict_num}.npy', features)
                left_x, top_y, right_x, bottom_y = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                y_coord = np.clip(top_y + bottom_y - top_y, 0, frame.size[1])
                x_coord = np.clip(left_x + right_x - left_x, 0, frame.size[0] - 1)

                img_to_save = Image.fromarray(
                    np.array(frame)[np.clip(top_y, 0, frame.size[1]):y_coord, np.clip(left_x, 0,
                                                                                      frame.size[
                                                                                          0]):x_coord])  # left_x, top_y, right_x, bottom_y    [y:y+h,x:x+w]
                img_to_save.save(f'{path_to_frame_faces}/{face_dict_num}.jpg')
        frames_tracked.append(frame_draw)

        frames_to_compute_sim.append(one_track_faces)

    height, width, layers = np.array(frame).shape

    frames_arrays = [np.array(frameee) for frameee in frames_tracked]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(frames_arrays, fps=fps)
    clip.write_videofile('my_video.mp4')
    if Path('my_video.mp4').exists():
        video_file = open('my_video.mp4', 'rb')
        video_bytes = video_file.read()

        st.video(video_bytes)

    model_name = 'enet_b2_8'
    fer = HSEmotionRecognizer(model_name=model_name, device='cuda')  # device is cpu or cuda

    path_to_video_splitted = Path(face_directory + '/' + file_name.stem)
    one_vid_folders = os_sorted([f for f in path_to_video_splitted.iterdir()])

    faces_paths = defaultdict(list)
    faces_occurences = defaultdict(int)
    for i in one_vid_folders:
        one_frame_faces = [*i.glob('*.jpg')]

        for face in one_frame_faces:
            faces_paths[face.name.split('.')[0]].append(face)
            faces_occurences[face.name.split('.')[0]] += 1
    exclude_ids = [*faces_paths.keys()]

    results = defaultdict(list)
    summed_values = np.zeros(8)

    for face_id in faces_paths:
        summed_values = np.zeros(8)

        one_id_faces = faces_paths[face_id]
        for id_face in one_id_faces:
            img = cv2.imread(str(id_face))
            emotion, scores = fer.predict_emotions(img, logits=False)
            summed_values = np.add(summed_values, scores)

            results[id_face.name.split('.')[0]] = summed_values

    argmax_emotion = [np.argmax(results[i]) for i in results]
    emotion_to_use = max(argmax_emotion, key=argmax_emotion.count)


    def get_emotion(results, emotion_id):
        least_emotion_score = {0: 0}
        emotion_results = defaultdict(list)
        for key in results:
            emotion_score = results[key][emotion_id]
            emotion_results[key] = emotion_score
            if emotion_score > [*least_emotion_score.values()][0]:
                least_emotion_score = {key: emotion_score}
        return least_emotion_score


    most_important_id = get_emotion(results, emotion_to_use)
    faces_to_show = faces_paths[[*most_important_id.keys()][0]]
    faces_to_show_str = [str(face) for face in faces_to_show]
    st.image(faces_to_show_str, width=100)
    st.write(id_to_emo[emotion_to_use])
