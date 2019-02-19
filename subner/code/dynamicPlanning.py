import numpy as np

class DynamicPlanning():
    def __init__(self):
        self.ad_len_weight = [0, 1.0/30, 1.0/30, 1.0/30, 1.0/30, 1.0/30, 1.0/30, 1.0/30]
        self.adp_prf_weight_para = [1.0, 0, 0]
        self.ad_ans_num_weight_para = 1.0
        self.adp_scores_weight_para = [1.0, 0]
        self.ans_dict_selected = None
        self.ans_dict_selected_info = None


    def select_ans_dict(self, data, ans_dict, feats_lists, maxlen=7):
        ans_dict_selected = []
        ans_dict_selected_info = {}
        for k, v in ans_dict.items():
            feature_length = k[1]
            feature_ans = list(k[2])
            if feature_length > maxlen:
                continue
            len_feature_ans = len(feature_ans)
            num_feature_ans = len(v[0][0])
            num_correct = 0
            num_correct_raw = 0
            for i in range(len(feats_lists)):
                feats_list = feats_lists[i]
                num_correct_add = len(feats_list) - feature_length + 1
                if num_correct_add < 0:
                    num_correct_add = 0
                num_correct += num_correct_add

            if num_correct == num_feature_ans:
                print('checked: {} len is {}'.format(k, num_correct))
                ans_dict_selected.append(k)
                info = []
                for adp_idx, adp in enumerate(v):
                    adp_scores = adp[0]
                    adp_scores_diff = [_[0] - _[1] for _ in adp_scores]
                    ad_abs_mean = np.mean(np.abs(adp_scores))
                    ad_abs_diff_mean = np.mean(np.abs(adp_scores_diff))
                    info.append([ad_abs_mean, ad_abs_diff_mean])
                ans_dict_selected_info[k] = info
            else:
                print('error in select_ans_dict: {}\n    num_correct:{}, num_feature_ans:{}'\
                      .format(k, num_correct, num_feature_ans))
        self.ans_dict_selected = ans_dict_selected
        self.ans_dict_selected_info = ans_dict_selected_info

    def label_id_to_substring_label_id(self, data, k, out=[10,11]):
        if k == 0:
            return 0
        if k in out:
            # if id is begin id or end id return -1
            return -1
        label_id_name = data.label_alphabet.get_instance(k)
        substring_label_id_name = label_id_name.split('-')[-1]
        id = data.substring_label_alphabet.instance2index[substring_label_id_name]
        return id

    def get_id_list(self, data, score_path, cur_label, last_label, len):
        if len < 2:
            print 'len should bigger than 2'
            return
        id_list = []
        id_list.append(self.label_id_to_substring_label_id(data, cur_label))
        id_list.append(self.label_id_to_substring_label_id(data, last_label))
        for i in range(2,len):
            last_label = score_path[-(i-1)][last_label]
            id_list.append(self.label_id_to_substring_label_id(data, last_label))
        id_list.reverse()
        return id_list

    def id_list_in_ad_ans(self, id_list, ad_ans):
        flag = False
        for i in id_list:
            if i in ad_ans:
                flag = True
                break
        return flag

    def reward(self, data, cur_label, num, ans_dict_selected, ans_dict_selected_info, ans_dict, feats_norm, pos, last_label, score_path):
        ## parameters
        ad_len_weight = self.ad_len_weight
        adp_prf_weight_para = self.adp_prf_weight_para
        ad_ans_num_weight_para = self.ad_ans_num_weight_para
        adp_scores_weight_para = self.adp_scores_weight_para

        id = self.label_id_to_substring_label_id(data, cur_label)
        reward_score = 0
        for ad_idx, ad in enumerate(ans_dict_selected):
            v = ans_dict[ad]
            ad_feature = ad[0]
            ad_len = ad[1]
            ad_ans = ad[2]
            ad_ans_num = len(ad_ans)
            if ad_len != 2:
                continue
            if ad_ans_num != 1:
                continue
            for adp_idx, adp in enumerate(v):
                # if adp_idx != 2:
                #     continue
                adp_scores = adp[0]
                adp_p = adp[1]
                adp_r = adp[2]
                adp_f = adp[3]
                ad_abs_mean = ans_dict_selected_info[ad][adp_idx][0]
                ad_abs_diff_mean = ans_dict_selected_info[ad][adp_idx][1]

                adp_prf_weight = adp_p * adp_prf_weight_para[0] + adp_r * adp_prf_weight_para[1]\
                                 + adp_f * adp_prf_weight_para[2]
                ad_ans_num_weight = 1 + (1.0 / ad_ans_num - 1.0) * ad_ans_num_weight_para


                # # use adp_idx == 2 and two line after this line. f:0.8631
                # if id in ad_ans:
                #     reward_score += adp_scores[num1][1] / 50.0

                # two line after this line. f:0.8632
                # if id in ad_ans and ad_len == 1:
                #     reward_score += adp_scores[num[1]][1] * (feats_norm / ad_abs_mean) * adp_p * (1.0 / ad_ans_num) / 30

                if id in ad_ans and ad_len == 1:
                    adp_scores_weight = adp_scores[num[1]][1] * (feats_norm / ad_abs_mean) * adp_scores_weight_para[0] \
                                        + (adp_scores[num[1]][1] - adp_scores[num[1]][0]) * (feats_norm / ad_abs_diff_mean) \
                                        * adp_scores_weight_para[1]
                    reward_score += adp_scores_weight * adp_prf_weight * ad_ans_num_weight * ad_len_weight[1]
                for len_i in range(2, 8):
                    if ad_len == len_i and ad_len_weight[len_i] != 0:
                        if pos+1 >= ad_len:
                            id_list = self.get_id_list(data, score_path, cur_label, last_label, len_i)
                            if self.id_list_in_ad_ans(id_list, ad_ans):
                                adp_scores_weight = adp_scores[num[len_i]][1] * (feats_norm / ad_abs_mean) \
                                                    * adp_scores_weight_para[0] + (adp_scores[num[len_i]][1] - adp_scores[num[len_i]][0]) \
                                                    * (feats_norm / ad_abs_diff_mean) * adp_scores_weight_para[1]
                                reward_score += adp_scores_weight * adp_prf_weight * ad_ans_num_weight * ad_len_weight[len_i]

        return reward_score

    def dynamic_planning_plus(self, data, feats_lists, inters, ans_dict):
        START_TAG = -2
        STOP_TAG = -1
        len_tag = 12
        ans_lists = []
        count = 1
        ans_dict_selected = self.ans_dict_selected
        ans_dict_selected_info = self.ans_dict_selected_info

        num = [0 for _ in range(8)]

        for feats in feats_lists:
            # if count%1000 == 0:
            #     print(count)
            count += 1
            score = []
            score_path = []
            back_path = []
            norm = [np.mean(np.abs(feats[i])) for i in range(len(feats))]
            score.append([inters[START_TAG][k] + feats[0][k] +
                          self.reward(data, k, num, ans_dict_selected, ans_dict_selected_info, ans_dict, norm[0], 0, -1, score_path)
                          for k in range(len_tag)])
            num[1] += 1
            score_path.append([k for k in range(len_tag)])
            for i in range(1, len(feats)):
                # score.append([max(score[-1][j] + inters[k][j] + feats[i][k] for j in range(len_tag)) for k in range(len_tag)])
                st = []
                stkpath = []
                for k in range(len_tag):
                    stk = []
                    for j in range(len_tag):
                        stk.append(score[-1][j] + inters[j][k] + feats[i][k] +
                                   self.reward(data, k, num, ans_dict_selected, ans_dict_selected_info, ans_dict, norm[i], i, j, score_path)
                          )
                    stkmax = max(stk)
                    st.append(stkmax)
                    stkpath.append(stk.index(stkmax))
                score.append(st)
                score_path.append(stkpath)

                for numi in range(8):
                    if i >= numi:
                        num[numi] += 1

            score_all = [score[-1][j] + inters[j][STOP_TAG] for j in range(len_tag)]
            highest = max(score_all)
            highest_path = score_all.index(highest)
            back_path.append(highest_path)
            for back in range(len(feats)-1, -1, -1):
                back_path.append(score_path[back][back_path[-1]])
            back_path = back_path[:-1]
            back_path.reverse()
            ans_lists.append(back_path)
        print('finally num is {}'.format(num))
        return ans_lists