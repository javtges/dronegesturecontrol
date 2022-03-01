import cv2
import numpy as np
import pdb
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor


def load_split_nvgesture(file_with_split ,list_split = list()):
    params_dictionary = dict()
    with open(file_with_split,'rt') as f:
          dict_name  = file_with_split[file_with_split.rfind('/')+1 :]
          dict_name  = dict_name[:dict_name.find('_')]

          for line in f:
            params = line.split(' ')
            params_dictionary = dict()

            params_dictionary['dataset'] = dict_name

            path = params[0].split(':')[1]
            for param in params[1:]:
                    parsed = param.split(':')
                    key = parsed[0]
                    # print(key)
                    if key == 'label':
                        # make label start from 0
                        label = int(parsed[1]) - 1 
                        params_dictionary['label'] = label
                    elif key in ('depth','color','duo_left'):
                        #othrwise only sensors format: <sensor name>:<folder>:<start frame>:<end frame>
                        sensor_name = key
                        #first store path
                        params_dictionary[key] = path + '/' + parsed[1]
                        #store start frame
                        params_dictionary[key+'_start'] = int(parsed[2])

                        params_dictionary[key+'_end'] = int(parsed[3])
                        # print(params_dictionary)
        
            # print(params_dictionary)
            params_dictionary['duo_right'] = params_dictionary['duo_left'].replace('duo_left', 'duo_right')
            params_dictionary['duo_right_start'] = params_dictionary['duo_left_start']
            params_dictionary['duo_right_end'] = params_dictionary['duo_left_end']          

            params_dictionary['duo_disparity'] = params_dictionary['duo_left'].replace('duo_left', 'duo_disparity')
            params_dictionary['duo_disparity_start'] = params_dictionary['duo_left_start']
            params_dictionary['duo_disparity_end'] = params_dictionary['duo_left_end']                  

            list_split.append(params_dictionary)
 
    return list_split

def load_data_from_file(example_config, sensor,image_width, image_height):

    path = "../nvGesture/" + example_config[sensor] + ".avi"
    start_frame = example_config[sensor+'_start']
    # print(start_frame)
    end_frame = example_config[sensor+'_end']
    # print(end_frame)
    label = example_config['label']

    frames_to_load = range(start_frame, end_frame)

    chnum = 3 if sensor == "color" else 1

    video_container = np.zeros((image_height, image_width, chnum, 80), dtype = np.uint8)

    cap = cv2.VideoCapture(path)

    ret = 1
    frNum = 0
    cap.set(1, start_frame);
    for indx, frameIndx in enumerate(frames_to_load):    
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame,(image_width, image_height))
            if sensor != "color":
                frame = frame[...,0]
                frame = frame[...,np.newaxis]
            video_container[..., indx] = frame
        else:
            print("Could not load frame")
            
    cap.release()
    # print("frames", len(frames_to_load))

    return video_container, label


class NVGesture():
        def __init__(self, data, target, transform=None):
            self.data = [torch.from_numpy(chunk).float() for chunk in data] # This will be the 1050*5 x 3 x 16 x 112 x 112 data?
            self.target = target#torch.FloatTensor(target).long #torch.from_numpy(target).long
            # print(self.data)
            # print(self.target)
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self,index):
            x = self.data[index]
            y = self.target[index]

            return x, y


def MakeDataloaders():
    
    sensors = ["color", "depth", "duo_left", "duo_right", "duo_disparity"]
    file_lists = dict()
    file_lists["test"] = "../nvGesture/nvgesture_test_correct_cvpr2016.lst"
    file_lists["train"] = "../nvGesture/nvgesture_train_correct_cvpr2016.lst"
    # file_lists["test"] = "./nvgesture_test_correct_cvpr2016.lst"
    # file_lists["train"] = "./nvgesture_train_correct_cvpr2016.lst"
    train_list = list()
    test_list = list()

    load_split_nvgesture(file_with_split = file_lists["train"],list_split = train_list)
    # print(train_list[0])
    print(len(train_list))
    load_split_nvgesture(file_with_split = file_lists["test"],list_split = test_list)
    # print(len(test_list))

    train_data = []
    test_data = []
    train_targets = []
    test_targets = []

    for train in range(len(train_list)): # len(train_list)
        data, label = load_data_from_file(example_config = train_list[train], sensor = sensors[0], image_width = 320, image_height = 240)
        # print(data.shape)
        # print(label)
        print(train)
        #print(train_list[train])
        data = data[64:176, 104:216, :, :]
        data2 = np.moveaxis(data,-1,0)
        # data3 = np.moveaxis(data2,-1,0)

        # print(data.shape)
        print(data2.shape)

        data_split = np.vsplit(data2,10) # split into 10 groups

        # print(len(data_split))
        # print(data_split[0].shape)

        for chunk in range(len(data_split)):
            train_data.append(data_split[chunk])
            train_targets.append(label)


    for test in range(len(test_list)): # len(test_list)
        data, label = load_data_from_file(example_config = test_list[test], sensor = sensors[0], image_width = 320, image_height = 240)
        # print(data.shape)
        # print(label)
        print(test)
        data = data[64:176, 104:216, :, :]
        data2 = np.moveaxis(data,-1,0)
        # data3 = np.moveaxis(data2,-1,0)

        # print(data.shape)
        print(data2.shape)

        data_split = np.vsplit(data2,10) # split into 10 groups

        # print(len(data_split))
        # print(data_split[0].shape)

        for chunk in range(len(data_split)):
            test_data.append(data_split[chunk])
            test_targets.append(label)

    # print(len(train_data))
    # print(train_data[0].shape)
    # print(train_targets)


    # CHANGE NUM_WORKERS AND EXPERIMENT!

    train_loader = torch.utils.data.DataLoader(dataset=NVGesture(train_data, train_targets), batch_size=350, shuffle=True, pin_memory=True, num_workers = 4)
    test_loader = torch.utils.data.DataLoader(dataset=NVGesture(test_data, test_targets), batch_size=350, shuffle=True, pin_memory=True, num_workers = 4)
    print("done loading data")

    return train_loader, test_loader




if __name__ == "__main__":

    # sensors = ["color", "depth", "duo_left", "duo_right", "duo_disparity"]
    # file_lists = dict()
    # file_lists["test"] = "./nvgesture_test_correct_cvpr2016.lst"
    # file_lists["train"] = "./nvgesture_train_correct_cvpr2016.lst"
    # train_list = list()
    # test_list = list()

    # load_split_nvgesture(file_with_split = file_lists["train"],list_split = train_list)
    # # print(train_list[0])
    # print(len(train_list))
    # load_split_nvgesture(file_with_split = file_lists["test"],list_split = test_list)
    # # print(len(test_list))

    # train_data = []
    # test_data = []
    # train_targets = []
    # test_targets = []

    # for train in range(1): # len(train_list)
    #     data, label = load_data_from_file(example_config = train_list[train], sensor = sensors[0], image_width = 320, image_height = 240)
    #     # print(data.shape)
    #     # print(label)
    #     data = data[64:176, 104:216, :, :]
    #     data2 = np.moveaxis(data,-1,0)
    #     data3 = np.moveaxis(data2,-1,0)

    #     # print(data.shape)
    #     # print(data3.shape)

    #     data_split = np.hsplit(data3,5) # split into 5 groups

    #     # print(len(data_split))
    #     # print(data_split[0].shape)

    #     for chunk in range(len(data_split)):
    #         train_data.append(data_split[chunk])
    #         train_targets.append(label)


    # for test in range(1): # len(test_list)
    #     data, label = load_data_from_file(example_config = train_list[test], sensor = sensors[0], image_width = 320, image_height = 240)
    #     # print(data.shape)
    #     # print(label)
    #     data = data[64:176, 104:216, :, :]
    #     data2 = np.moveaxis(data,-1,0)
    #     data3 = np.moveaxis(data2,-1,0)

    #     # print(data.shape)
    #     # print(data3.shape)

    #     data_split = np.hsplit(data3,5) # split into 5 groups

    #     # print(len(data_split))
    #     # print(data_split[0].shape)

    #     for chunk in range(len(data_split)):
    #         test_data.append(data_split[chunk])
    #         test_targets.append(label)

    # print(len(train_data))
    # print(train_data[0].shape)
    # print(train_targets)


    # train_loader = torch.utils.data.DataLoader(dataset=NVGesture(train_data, train_targets), batch_size=1, shuffle=False)

    train_loader, test_loader = MakeDataloaders()
    print(train_loader)
    # train_features, train_labels = next(iter(train_loader))
    # print(train_labels)

    
    # This shows what the 80 - frame videos are doing!
    # for i in range(80):
        # frame = data[i,:,:,:]
        # cv2.imshow("frame", frame)
        # cv2.waitKey(50)