# test_functions.py

import socket
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import requests

from functions import (
    bgr2hex,
    compute_affine,
    draw_edge_locs,
    hex2bgr,
    is_port_open,
    obj_rect2coords,
    wait_for_service,
)

# ------------------------------------------------------------------------


def test_compute_affine():
    obj_definition = {
        "top": 10,
        "left": 20,
        "width": 30,
        "height": 40,
        "scaleX": 1,
        "scaleY": 1,
        "angle": 45,
    }
    dsize = (100, 100)
    frame_affine_matrix = np.eye(3)
    result = compute_affine(obj_definition, dsize, frame_affine_matrix)
    assert result.shape == (3, 3)


def test_draw_edge_locs():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    edge_locs = [(10, 10), (20, 20)]
    result = draw_edge_locs(frame, edge_locs)
    assert np.any(result)


def test_hex2bgr():
    hex_color = "#FF0000"
    result = hex2bgr(hex_color)
    assert result == (0, 0, 255)


def test_bgr2hex():
    bgr_color = (0, 0, 255)
    result = bgr2hex(bgr_color)
    assert result == "#FF0000"


def test_is_port_open():
    # You may need to adjust the IP and port for your environment
    ip = "127.0.0.1"
    port = 80
    result = is_port_open(ip, port)
    assert result == False  # Assuming the port is open


def test_obj_rect2coords():
    obj_definition = {
        "top": 10,
        "left": 20,
        "width": 30,
        "height": 40,
        "scaleX": 1,
        "scaleY": 1,
        "angle": 45,
    }
    result = obj_rect2coords(obj_definition)
    assert result == (10, 20, 30, 40, 45)


def test_wait_for_service(monkeypatch):
    # Mock the requests.get method
    class MockResponse:
        def __init__(self, status_code):
            self.status_code = status_code

        def raise_for_status(self):
            pass

    def mock_get(*args, **kwargs):
        return MockResponse(200)

    monkeypatch.setattr(requests, "get", mock_get)

    # Call the function and assert
    wait_for_service()
    # If no exception is raised, the test passes
