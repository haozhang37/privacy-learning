import math
import random

# 样本数
num = 10000
#精确度
accuracy = 1000

def Cos_generator():
    values = []
    total = 0
    for i in range(90):
        a = math.cos(i * math.pi / 180)
        a *= accuracy
        values.append(a)
        total += a
    return values, total


def random_index(values, total):
    start = 0
    index = 0
    randnum = random.randint(1, total)

    for index, scope in enumerate(values):
        start += scope
        if randnum < start:
            break
    return index


def Sphere_Sampling(n):
    values, total = Cos_generator()
    samples = []
    # print("total %f" % total)
    for i in range(n):
        phi = random_index(values, int(total) + 1)
        theta = random.randint(0, 359)
        rotate = random.randint(0, 359)
        samples.append([phi, theta, rotate])
        #print("[%d, %d]" % (phi, theta))
    return samples


def sample_check(values, total, samples):
    count = [0 for i in range(90)]
    for i, angles in enumerate(samples):
        count[angles[0]] += 1
    for i, cos in enumerate(values):
        count[i] /= (num/total)*cos
        print("%f" % count[i])


def main():
    values, total = Cos_generator()
    samples = Sphere_Sampling(num)
    sample_check(values, total, samples)


if __name__ == "__main__":
    main()
