# Grading script given to learners, this file should work in both Python 2 and 3

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt


def l2_squared(v, w):
    """
    Compute the squared distance between 2 1D vectors v and w
    Output D is a NxM matrix, where N = dim(v) and M = dim(w)
    D(n,m) = (v_n - w_m)^2
    """
    # shape v into a column, w into a row
    v = v.reshape([-1, 1])
    w = w.reshape([1, -1])

    # duplicate v and w to vectorize distance calculation
    v_mat = np.tile(v, w.shape[0])
    w_mat = np.tile(w, (v.shape[0], 1))

    # compute the distance and return
    D = (v_mat - w_mat) ** 2
    return D


def grade(waypoints, solution, visualize=False):
    """
    Compares a set of solution data to waypoint data
    solution is a Mx4 matrix, each row contains [x,y,z,v]
    waypoints is a Nx4 matrix, each row contains [x,y,z,v]

    Solution passes if:
    the path is close to waypoints
    speed at the points closest to the path are close to waypoint speeds
    pass conditions are set by a distance and speed threshold and % of
    waypoints correct

    Returns a dictionary containing the solution, waypoints, and performance
    performance contains:
        nearest distance to waypoints
        index of points on solution path closest to waypoints
        speeds at the closest points
        whether the solution passes or fails

    Visualizing the solution shows the solution path and speed at waypoints
    """
    N = waypoints.shape[0]

    results = {
        "d_thresh": 3,  # distance to waypoints
        "v_thresh": 3,  # speed difference at waypoints
        "pass_percentage": 60,  # percentage of correct waypoints required
        "solution": solution,
        "waypoints": waypoints,
        "dists2": np.zeros((N, 1)),
        "vels": np.zeros((N, 1)),
        "inds": np.zeros((N, 1), dtype="int32"),
        "pass_dist": True,
        "pass_vel": True,
    }

    # compute the distance from the waypoints to the solution
    results["dists2"] = (
        l2_squared(waypoints[:, 0], solution[:, 0])
        + l2_squared(waypoints[:, 1], solution[:, 1])
        + l2_squared(waypoints[:, 2], solution[:, 2])
    )

    results["inds"] = np.argmin(results["dists2"], 1)
    results["dists2"] = np.min(results["dists2"], 1)
    results["vels"] = solution[results["inds"], 3].reshape([-1, 1])

    # check the distance and speed at waypoints, find path locations that satisfy both
    dists_correct = results["dists2"] ** 0.5 <= results["d_thresh"]
    loss = np.abs(results["vels"][dists_correct, 0] - waypoints[dists_correct, 3])
    vels_correct = loss <= results["v_thresh"]
    percent_correct = (np.sum(vels_correct) / N) * 100

    if percent_correct < results["pass_percentage"]:
        print(
            "Assessment failed, only {0:.2f}% of waypoints completed. {1:.2f}% required to pass".format(
                percent_correct, results["pass_percentage"]
            )
        )
    else:
        print(
            "Assessment passed! {0:.2f}% of waypoints completed.".format(
                percent_correct
            )
        )

    if visualize:
        display_path(results)

    return results


def display_path(results):
    """
    Generates 2 plots from the waypoints and solution

    1) Plot the path and waypoints, show distance tolerance circles
    2) Plot speed at waypoints and path speed, show speed tolerance bounds
    """

    fig, ax = plt.subplots()
    plt.plot(
        results["waypoints"][:, 0], results["waypoints"][:, 1], "*", label="waypoints"
    )
    plt.plot(results["solution"][:, 0], results["solution"][:, 1], label="solution")

    plt.legend()
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Waypoints and Solution Path")
    plt.savefig("result1.png")

    plt.subplots()
    plt.plot(results["waypoints"][:, 3], label="reference speed")
    plt.plot(results["solution"][results["inds"], 3], label="vehicle speed")
    plt.plot(results["waypoints"][:, 3] + results["v_thresh"], "--", color="C2")
    plt.plot(results["waypoints"][:, 3] - results["v_thresh"], "--", color="C2")

    plt.legend()
    plt.xlabel("Waypoint #")
    plt.ylabel("Speed (km/h)")
    plt.title("Speed Profiles")

    plt.savefig("result2.png")


if __name__ == "__main__":

    waypoint_file = "./waypoints.txt"
    waypoints = np.genfromtxt(waypoint_file, delimiter=" ")
    refined_waypoints = waypoints[5:]
    solution_file = "./solution.txt"
    solution = np.genfromtxt(solution_file, delimiter=" ")

    results = grade(refined_waypoints, solution, True)
