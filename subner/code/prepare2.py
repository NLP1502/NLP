def classification(data, feature_name, feature_length, class_num, feature_ans):
    print "Training classification..."

    # feature_name = 'word'
    # feature_length = 3
    # class_num = 2
    # feature_ans = [2,3]

    feature_ans_new = []
    for fa in feature_ans.split(','):
        feature_ans_new.append(data.substring_label_alphabet.instance2index[fa])
    feature_ans = feature_ans_new

    feature_name_id = data.substring_names.index(feature_name)

    model = LSTMText(data, feature_name, feature_ans, feature_length)
    print "model:{}".format(model)

    if data.gpu:
        model.cuda()

    if data.optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=data.lr, momentum=data.momentum, weight_decay=data.l2)
        if data.use_mapping:
            optimizer_wc = optim.SGD(model.word_hidden.wordrep.w.parameters(), lr=data.lr, momentum=data.momentum,
                                     weight_decay=data.l2)
    elif data.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=data.lr, weight_decay=data.l2)
        if data.use_mapping:
            optimizer_wc = optim.Adam(model.word_hidden.wordrep.w.parameters(), lr=data.lr, weight_decay=data.l2)
    else:
        print("Optimizer illegal: %s , use sgd or adam." % data.optimizer)
        exit(0)

    best_dev = -10
    best_dev_epoch = -1
    best_test = -10
    best_test_epoch = -1
    loss_function = torch.nn.MultiLabelSoftMarginLoss()
    # start training
    for idx in range(data.iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch: %s/%s" % (idx + 1, data.iteration))
        if data.optimizer == "SGD":
            optimizer = lr_decay(optimizer, idx, data.lr_decay, data.lr)

        pt = 0
        nt = 0
        pf = 0
        nf = 0
        b = 0.00000001

        instance_count = 0

        sample_loss = 0
        sample_mapping_loss = 0

        total_loss = 0
        total_mapping_loss = 0

        right_token = 0
        whole_token = 0
        whole_token_per_check = 0

        random.shuffle(data.substring_train_Ids[feature_name_id][feature_length])

        # set model in train mode
        model.train()
        model.zero_grad()
        batch_size = data.batch_size
        batch_id = 0
        train_num = len(data.substring_train_Ids[feature_name_id][feature_length])
        total_batch = train_num // batch_size + 1

        for batch_id in range(total_batch):
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            if end > train_num:
                end = train_num
            whole_token_per_check += end - start

            ### !!!!!!!!!!
            instance = data.substring_train_Ids[feature_name_id][feature_length][start:end]



            if not instance:
                continue
            label_class = 0
            batch_word, batch_wordlen, batch_wordrecover, batch_label, mask = \
                batchify_with_label_classification(instance, class_num, feature_ans, data.gpu)
            instance_count += 1
            score = model(batch_word, batch_wordlen, mask)

            weight = 1
            loss = loss_function(score, weight * batch_label.float())

            total_loss += loss.data

            predict = score.data.topk(1, dim=1)[1].cpu().tolist()
            true_target = batch_label.data.float().topk(1, dim=1)[1].cpu().tolist()

            predict_label_and_marked_label_list = []
            for jj in range(batch_label.size(0)):
                if jj < end - start:
                    predict_label_and_marked_label_list.append((predict[jj], true_target[jj]))

            for pre, tru in predict_label_and_marked_label_list:
                if tru[0] == 0:
                    if pre[0] == 0:
                        nf += 1
                    else:
                        pf += 1
                else:
                    if pre[0] == 0:
                        nt += 1
                    else:
                        pt += 1

            if batch_id % data.show_loss_per_batch == 0:
                p = pt / (pt + pf + b)
                r = pt / (pt + nt + b)
                a = (pt + nf) / (whole_token_per_check + b)
                f = 2 * p * r / (p + r + b)
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                print("     Instance: %s; Time: %.2fs; a: %.4f; p: %.4f; r: %.4f; f: %.4f" % (
                    batch_id, temp_cost,
                    a, p, r, f))
                whole_token_per_check = 0
                pt = 0
                nt = 0
                pf = 0
                nf = 0
                sys.stdout.flush()

            loss.backward()
            if data.clip != None:
                torch.nn.utils.clip_grad_norm(model.parameters(), data.clip)
            optimizer.step()
            model.zero_grad()

        temp_time = time.time()
        temp_cost = temp_time - temp_start

        print("     Instance: %s; Time: %.2fs; a: %.4f; p: %.4f; r: %.4f; f: %.4f" % (
            batch_id, temp_cost,
            a, p, r, f))
        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start

        print(
                "Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s" % (
            idx + 1, epoch_cost, train_num / epoch_cost, total_loss))


        # continue
        speed, acc, p, r, f = evaluate_classification(data, model, "dev",
                feature_name, feature_length, feature_ans)
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish


        current_score = f
        print("Dev: time: %.2fs speed: %.2fst/s; acc: %.4f" % (dev_cost, speed, acc))

        save_when_get_new_best_score_on_dev = False
        save_each_epoch = True

        if save_when_get_new_best_score_on_dev:
            if current_score > best_dev:
                if data.seg:
                    print "Exceed previous best f score:", best_dev
                else:
                    print "Exceed previous best acc score:", best_dev
                if data.save_model:
                    model_name = data.model_dir + data.state_training_name + '.'\
                                 + feature_name + 'len%d'%(feature_length) + '.ans'\
                                 + '-'.join([str(_) for _ in feature_ans]) + '.score' + str(current_score)[2:-1]
                    print "Save current best model in file:", model_name
                    torch.save(model.state_dict(), model_name)
                best_dev = current_score
                best_dev_epoch = idx

        if save_each_epoch:
            if current_score > best_dev:
                best_dev = current_score
                best_dev_epoch = idx
            if data.seg:
                print "Exceed previous best f score:", best_dev
            else:
                print "Exceed previous best acc score:", best_dev
            if data.save_model:
                model_name = data.model_dir + data.state_training_name + '.'\
                             + feature_name + 'len%d'%(feature_length) + '.ans'\
                             + '-'.join([str(_) for _ in feature_ans]) + '.score' + str(current_score)[2:-1]
                print "Save model at %d epoch in file:" % (idx), model_name
                torch.save(model.state_dict(), model_name)

        ## decode test
        speed, acc, p, r, f = evaluate_classification(data, model, "test",
            feature_name, feature_length, feature_ans)

        if f > best_test:
            best_test = f
            best_test_epoch = idx

        test_finish = time.time()
        test_cost = test_finish - dev_finish
        if data.seg:
            print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
                test_cost, speed, acc, p, r, f))
        else:
            print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f" % (test_cost, speed, acc))
        print('best_dev_score: %.4f, best_dev_epoch:%d'%(best_dev,best_dev_epoch))
        print('best_test_score: %.4f, best_test_epoch:%d'%(best_test,best_test_epoch))
        gc.collect()