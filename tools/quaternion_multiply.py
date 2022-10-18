def quaternion_multiply(q,p):
    s = q[0] * p[0] - q[1] * p[1] - q[2] * p[2] - q[3] * p[3]
    x = q[0] * p[1] + q[1] * p[0] + q[2] * p[3] - q[3] * p[2]
    y = q[0] * p[2] - q[1] * p[3] + q[2] * p[0] + q[3] * p[1]
    z = q[0] * p[3] + q[1] * p[2] - q[2] * p[1] + q[3] * p[0]
    result = [s,x,y,z]
    return result