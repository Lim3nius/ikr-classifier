#!/usr/bin/env python2

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import rand
from numpy.linalg import eigh, det, inv, solve, norm
from mpl_toolkits.mplot3d import Axes3D
from ikrlib import *


def main():
    dirs = ['data/target_train', 'data/non_target_train', 'data/target_dev', 'data/non_target_dev']
    dat1 = png2fea(dirs[0])
    dat2 = png2fea(dirs[1])
    dat3 = png2fea(dirs[2])
    dat4 = png2fea(dirs[3])
    dat5 = png2fea('data/eval')
    dat1.update(dat3)
    dat2.update(dat4)

    dat1 = np.vstack([d.flatten() for d in dat1.values()])
    dat2 = np.vstack([d.flatten() for d in dat2.values()])
    dat3 = np.vstack([d.flatten() for d in dat3.values()])
    dat4 = np.vstack([d.flatten() for d in dat4.values()])
    
    """
    # duplicate target data to same amount as non-target
    new_dat = dat1.copy()
    for _ in range(6):
        new_dat = np.r_[new_dat, dat1]
    dat1 = new_dat
    """

    print "Computing cov..."
    cov = np.cov(np.vstack([dat1, dat2]).T, bias=True)
    dim = cov.shape[1]

    print "Computing eigh..."
    d,e = scipy.linalg.eigh(cov, eigvals=(dim-3, dim-1), turbo=True)

    print "Computing dot..."
    target = dat1.dot(e)
    non_target = dat2.dot(e)
    test_target = dat3.dot(e)
    test_non_target = dat4.dot(e)

    eval_data = {}
    for k, v in dat5.iteritems():
        eval_data[k.replace(".png", "")] = v.flatten().dot(e)

    print "Training classifier..."
    x = np.r_[target, non_target]
    t = np.r_[np.ones(len(target)), np.zeros(len(non_target))]
    w, w0, _ = train_generative_linear_classifier(x, t)

    for _ in range(50):
        w, w0 = train_linear_logistic_regression(x, t, w, w0)

    """
    # plot 3D
    ax = plt.figure().add_subplot(111,projection='3d')
    ax.scatter(target[:,2], target[:,1],target[:,0], c='r', marker='o')
    ax.scatter(non_target[:,2], non_target[:,1], non_target[:,0], c='b', marker='^')
    ax.scatter(test_target[:,2], test_target[:,1],test_target[:,0], c='m', marker='o')
    ax.scatter(test_non_target[:,2], test_non_target[:,1], test_non_target[:,0], c='c', marker='^')
    plt.legend()
    plt.show()

    # plot 2D
    plt.plot(target[:,0], target[:,1], 'r.', label="Target_train")
    plt.plot(non_target[:,0], non_target[:,1], 'b.', label="Non_target_train")
    plt.plot(test_target[:,0], test_target[:,1], 'm.', label="Target_dev")
    plt.plot(test_non_target[:,0], test_non_target[:,1], 'c.', label="Non_target_dev")
    plt.plot(eval_data[:,0], eval_data[:,1], 'g.', label="Eval_data")
    x1, x2 = plt.axis()[:2]
    y1 = (-2 * w0 - (w[0] * x1)) / w[1]
    y2 = (-2 * w0 - (w[0] * x2)) / w[1]
    plt.plot([x1, x2], [y1, y2], 'k', linewidth=2)
    plt.legend()
    plt.show()
    """
    
    """
    print "Test targets:"
    target_sum = 0
    for tgt in test_target:
        res = w.dot(tgt) + w0
        res -= np.log(dat1.shape[0] / float(dat2.shape[0]))
        target_sum += res
        print res

    print "Test non_targets:"
    non_target_sum = 0
    for tgt in test_non_target:
        res = w.dot(tgt) + w0
        res -= np.log(dat1.shape[0] / float(dat2.shape[0]))
        non_target_sum += res
        print res

    print "Target sum:", target_sum
    print "Non_target sum:", non_target_sum
    """

    out_file = open("image_linear_results.txt", "w")
    hit_count = 0
    for k, v in eval_data.iteritems():
        res = w.dot(v) + w0
        #res -= np.log(dat1.shape[0] / float(dat2.shape[0]))
        if res > 0:
            hit_count += 1
            print k, res, "1"
            out_file.write(k + " " + str(res) + " 1\n")
        else:
            print k, res, "0"
            out_file.write(k + " " + str(res) + " 0\n")

    out_file.close()
    print "Hit count:", hit_count


if __name__ == '__main__':
    main()
