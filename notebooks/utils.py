import numpy as np
import struct
import time
import math
import json
from pathlib import Path
from tqdm.auto import tqdm
import os

base_dir = str(Path(__file__).resolve().parents[1]) + "/"

json_dir = base_dir + "jsons/"
optuna_dir = base_dir + "optuna_databases/"
firmware_dir = base_dir + "firmware/"
data_dir = base_dir + "data/"
    
class CW_Agent_tiny_ml:
    def __init__(self, target):
        '''
        ChipWhisperer TinyML Agent. Uses the SimpleSerial 2 (SS_VER_2_1) protocol
        
        :param: target - chipwhisperer target used for serial communication
        '''  
        self.target = target
        self.clear_serial()

    def clear_serial(self):
        self.target.flush()

    def reset(self, scope):
        scope.io.nrst = 'low'
        time.sleep(0.25)
        scope.io.nrst = 'high_z'
        time.sleep(0.25)
        self.clear_serial()
        
    def reboot(self, scope):
        self.reset(scope)
        
    def parse_usps_line(self, line):
        if not line.strip():
            return None
        parts = line.strip().split()
        try:
            # Extract the label (first element)
            label = int(parts[0])
            # Extract the feature values
            values = [float(pair.split(":")[1]) for pair in parts[1:]]
        except (IndexError, ValueError):
            return None  # Skip malformed lines
        return label, np.array(values)

    def bytes_to_float(self, byte_list):
        # Convert string byte representations to integer byte values
        byte_array = bytearray(ord(b) for b in byte_list)
        
        # Unpack the byte array into a float
        float_value, = struct.unpack('f', byte_array)
        return float_value

    ########################################
    # Fastgrnn
    def send_fault_no_and_cmd(self, fault_no_1, fault_no_2, select_image_cmd, actual_no, unprotected_protected_cmd):
        self.clear_serial()
        self.target.send_cmd(unprotected_protected_cmd, 0x01, int.to_bytes(fault_no_1, 1, 'little') +
                             int.to_bytes(fault_no_2, 1, 'little') +
                             int.to_bytes(select_image_cmd, 1, 'little') +
                             int.to_bytes(actual_no, 1, 'little'))
        # Settle time: let the target finish processing this 0x01 command before
        # the caller streams the image. Without it the first 0x02 image packet
        # arrives while the target is still busy and is dropped (shifting the
        # image). This is what lets send_input_image skip the read-back verify.
        time.sleep(0.02)

    def send_image(self, image_data, unprotected_protected_cmd):
        """Stream one 16×16 image in 5 packets of 62 floats (248 bytes each)."""
        self.clear_serial()
        CHUNK_SIZE = 62    

        # if this is a single 1-D image, wrap it into a list
        if isinstance(image_data, np.ndarray) and image_data.ndim == 1:
            image_data = [image_data]

        for image in image_data:
            # send all chunks of this one image
            for offset in range(0, 256, CHUNK_SIZE):
                sub = image[offset : offset + CHUNK_SIZE]
                float_bytes = struct.pack('<%df' % len(sub), *sub)
                # scmd=0x02 means “image data”
                self.target.send_cmd(unprotected_protected_cmd, 0x02, float_bytes)
                # throttle a bit so the SAM4S can keep up
                time.sleep(0.005)

    def run_fast_grnn(self, unprotected_protected_cmd):
        self.clear_serial()
        self.target.send_cmd(unprotected_protected_cmd, 0x04, bytearray([]))
        time.sleep(1)
        return self.target.simpleserial_read(cmd='p', pay_len=1, timeout=0)[0]

    def run_fast_grnn_without_return(self, unprotected_protected_cmd):
        self.clear_serial()
        self.target.send_cmd(unprotected_protected_cmd, 0x04, bytearray([]))
        time.sleep(1)
        
    def predict_fast_grnn(self, image: np.ndarray, unprotected_protected_cmd) -> int:
        '''
        High-level API: clear once, send image, infer, and return label.
        '''
        self.send_image(image, unprotected_protected_cmd)
        return self.run_fast_grnn(unprotected_protected_cmd)
        
    def predict_fast_grnn_without_return(self, image: np.ndarray, unprotected_protected_cmd):
        '''
        High-level API: clear once, send image, infer, and return label.
        '''
        self.send_image(image, unprotected_protected_cmd)
        self.run_fast_grnn_without_return(unprotected_protected_cmd)

    def get_matvec_output(self, unprotected_protected_cmd):
        HIDDEN_DIMS = 32
        self.clear_serial()
        self.target.send_cmd(unprotected_protected_cmd, 0x10, bytearray([]))
        # Read back the 32 floats = 128 bytes
        matvec = self.target.simpleserial_read(cmd='m', pay_len=HIDDEN_DIMS*4, timeout=2000)
        # Unpack little-endian floats
        matvec = np.array(struct.unpack('<32f', bytes(matvec)), dtype=np.float32)
        return matvec

    def get_hidden_output(self, unprotected_protected_cmd):
        HIDDEN_DIMS = 32
        self.clear_serial()
        self.target.send_cmd(unprotected_protected_cmd, 0x20, bytearray([]))
        # Read back the 32 floats = 128 bytes
        hidden = self.target.simpleserial_read(cmd='h', pay_len=HIDDEN_DIMS*4, timeout=2000)
        # Unpack little-endian floats
        hidden = np.array(struct.unpack('<32f', bytes(hidden)), dtype=np.float32)
        return hidden

    def get_class_score_output(self, unprotected_protected_cmd):
        NUM_CLASSES = 10
        self.clear_serial()
        self.target.send_cmd(unprotected_protected_cmd, 0x40, bytearray([]))
        # Read back the 32 floats = 128 bytes
        class_scores = self.target.simpleserial_read(cmd='c', pay_len=NUM_CLASSES*4, timeout=2000)
        # Unpack little-endian floats
        class_scores = np.array(struct.unpack('<10f', bytes(class_scores)), dtype=np.float32)
        return class_scores


    ########################################
    # Logistic Regression
    def send_fault_no(self, fault_no_1, fault_no_2, unprotected_protected_cmd):
        self.clear_serial()
        self.target.send_cmd(unprotected_protected_cmd, 0x01, int.to_bytes(fault_no_1, 1, 'little') + 
                             int.to_bytes(fault_no_2, 1, 'little'))

    def _send_input_data_raw(self, input_data, unprotected_protected_cmd):
        """Stream one or more 1-D inputs in packets of 11 doubles each (no verification)."""
        self.clear_serial()
        CHUNK_SIZE = 11  # number of doubles per packet

        # normalize to a list of 1-D arrays
        if isinstance(input_data, np.ndarray) and input_data.ndim == 1:
            input_data = [input_data]

        for data in input_data:
            # send all chunks of this one image
            for offset in range(0, len(data), CHUNK_SIZE):
                sub = data[offset : offset + CHUNK_SIZE]
                # pack as little-endian doubles
                double_bytes = struct.pack('<%dd' % len(sub), *sub)
                # scmd=0x02 means “image data”
                self.target.send_cmd(unprotected_protected_cmd, 0x02, double_bytes)
                # throttle so the SAM4S can keep up
                time.sleep(0.005)

    def send_input_data(self, input_data, unprotected_protected_cmd, verify=True, max_retries=5):
        """Stream one 1-D input (11 doubles) AND verify it actually landed on the device.

        Glitch campaigns can desync the SimpleSerial link / leave the target mid-reset,
        which silently drops the 0x02 input packet so inference runs on a zero buffer.
        We read the device's input buffer back (scmd 0x20) and, on mismatch, resync the
        framing (reset_comms) and resend, up to `max_retries` times.

        Returns True if the input was confirmed on-device (or verification disabled).
        """
        # only a single 1-D vector can be verified against the 11-double echo
        expected = None
        if isinstance(input_data, np.ndarray) and input_data.ndim == 1:
            expected = input_data.astype(np.float64)

        self._send_input_data_raw(input_data, unprotected_protected_cmd)
        if not verify or expected is None:
            return True

        for _ in range(max_retries):
            try:
                echo = self.get_input_echo(unprotected_protected_cmd)
                if echo is not None and np.allclose(echo, expected, rtol=1e-6, atol=1e-9):
                    return True
            except Exception:
                pass
            # resync the SS2 framing (flush any partial/garbled frame), then resend
            try:
                self.target.reset_comms()
            except Exception:
                pass
            self._send_input_data_raw(input_data, unprotected_protected_cmd)

        print("[send_input_data] WARNING: could not confirm input on device after "
              f"{max_retries} retries (is the new firmware with scmd 0x20 flashed?)")
        return False

    def get_input_echo(self, unprotected_protected_cmd):
        """Read back the 11 doubles currently in the device input buffer (debug scmd 0x20)."""
        N = 11
        self.clear_serial()
        self.target.send_cmd(unprotected_protected_cmd, 0x20, bytearray([]))
        raw = self.target.simpleserial_read(cmd='n', pay_len=N * 8, timeout=2000)
        return np.array(struct.unpack('<%dd' % N, bytes(raw)), dtype=np.float64)

    def run_logistic_regression(self, unprotected_protected_cmd):
        self.clear_serial()
        self.target.send_cmd(unprotected_protected_cmd, 0x04, bytearray([]))
        time.sleep(1)
        return self.target.simpleserial_read(cmd='p', pay_len=1, timeout=0)[0]
        
    def run_logistic_regression_without_return(self, unprotected_protected_cmd):
        self.clear_serial()
        self.target.send_cmd(unprotected_protected_cmd, 0x04, bytearray([]))
        time.sleep(1)

    def predict_logistic_regression(self, input_data, unprotected_protected_cmd):
        self.send_input_data(input_data, unprotected_protected_cmd)
        return self.run_logistic_regression(unprotected_protected_cmd)

    def predict_logistic_regression_without_return(self, input_data, unprotected_protected_cmd):
        self.send_input_data(input_data, unprotected_protected_cmd)
        self.run_logistic_regression_without_return(unprotected_protected_cmd)

    def get_products_output(self, unprotected_protected_cmd):
        """Per-feature contributions X[j]*w[j] (the 11 terms summed into the logit). 11 doubles."""
        N = 11
        self.clear_serial()
        self.target.send_cmd(unprotected_protected_cmd, 0x10, bytearray([]))
        raw = self.target.simpleserial_read(cmd='m', pay_len=N * 8, timeout=2000)
        # Unpack little-endian doubles (LR computes in double throughout)
        return np.array(struct.unpack('<%dd' % N, bytes(raw)), dtype=np.float64)

    ########################################
    # Wakeword
    def send_input_data_wwd(self, input_data, unprotected_protected_cmd):
        """Stream one or more 1-D inputs in packets of 40 floats each."""
        self.clear_serial()
        CHUNK_SIZE = 40   

        # if this is a single 1-D input, wrap it into a list
        if isinstance(input_data, np.ndarray) and input_data.ndim == 1:
            input_data = [input_data]
        
        for data in input_data:
            # send all chunks of this one image
            for offset in range(0, len(data), CHUNK_SIZE):
                sub = data[offset : offset + CHUNK_SIZE]
                float_bytes = struct.pack('<%df' % len(sub), *sub)
                # scmd=0x02 means “image data”
                self.target.send_cmd(unprotected_protected_cmd, 0x02, float_bytes)
                # throttle a bit so the SAM4S can keep up
                time.sleep(0.005)

    def run_wake_word(self, unprotected_protected_cmd):
        self.clear_serial()
        self.target.send_cmd(unprotected_protected_cmd, 0x04, bytearray([]))
        time.sleep(1)
        return self.target.simpleserial_read(cmd='p', pay_len=1, timeout=0)[0]

    def run_wake_word_without_return(self, unprotected_protected_cmd):
        self.clear_serial()
        self.target.send_cmd(unprotected_protected_cmd, 0x04, bytearray([]))
        time.sleep(1)
    
    def predict_wake_word(self, input_data, unprotected_protected_cmd):
        self.send_input_data_wwd(input_data, unprotected_protected_cmd)
        return self.run_wake_word(unprotected_protected_cmd)

    def predict_wake_word_without_return(self, input_data, unprotected_protected_cmd):
        self.send_input_data_wwd(input_data, unprotected_protected_cmd)
        self.run_wake_word(unprotected_protected_cmd)

    def get_fc1_output(self, unprotected_protected_cmd):
        TOTAL    = 256
        CHUNK    = 128
        n_chunks = TOTAL // CHUNK  
    
        raw = bytearray()
        for i in range(n_chunks):
            self.target.send_cmd(unprotected_protected_cmd, 0x08, bytearray([]))
            chunk = self.target.simpleserial_read('a', pay_len=CHUNK, timeout=2000)
            if chunk is None:
                raise IOError(f"Timeout reading chunk {i}: expected {CHUNK} bytes")
            raw.extend(chunk)
    
        if len(raw) != TOTAL:
            raise IOError(f"Expected {TOTAL} bytes, got {len(raw)}")
    
        # unpack as 256 signed 8-bit ints
        vals = struct.unpack(f'<{TOTAL}b', bytes(raw))
        return np.array(vals, dtype=np.int8)

    def get_fc2_output(self, unprotected_protected_cmd):
        TOTAL    = 256
        CHUNK    = 128
        n_chunks = TOTAL // CHUNK  
    
        raw = bytearray()
        for i in range(n_chunks):
            self.target.send_cmd(unprotected_protected_cmd, 0x10, bytearray([]))
            chunk = self.target.simpleserial_read('b', pay_len=CHUNK, timeout=2000)
            if chunk is None:
                raise IOError(f"Timeout reading chunk {i}: expected {CHUNK} bytes")
            raw.extend(chunk)
    
        if len(raw) != TOTAL:
            raise IOError(f"Expected {TOTAL} bytes, got {len(raw)}")
    
        # unpack as 256 signed 8-bit ints
        vals = struct.unpack(f'<{TOTAL}b', bytes(raw))
        return np.array(vals, dtype=np.int8)

    def get_fc3_output(self, unprotected_protected_cmd):
        WWD_OUTPUT_DIM = 2
        self.clear_serial()
        self.target.send_cmd(unprotected_protected_cmd, 0x20, bytearray([]))
        # Read back the 2 signed‐bytes = 2 bytes
        raw = self.target.simpleserial_read(cmd='c', pay_len=WWD_OUTPUT_DIM, timeout=2000)
        # Unpack little‐endian signed chars ('b')
        fc3_output = struct.unpack(f'<{WWD_OUTPUT_DIM}b', bytes(raw))
        # Convert to a NumPy array of dtype int8 (range is -128..127)
        return np.array(fc3_output, dtype=np.int8)

    def get_softmax_output(self, unprotected_protected_cmd):
        WWD_OUTPUT_DIM = 2
        self.clear_serial()
        self.target.send_cmd(unprotected_protected_cmd, 0x40, bytearray([]))
        # Read back the 2 signed‐bytes = 2 bytes
        raw = self.target.simpleserial_read(cmd='d', pay_len=WWD_OUTPUT_DIM, timeout=2000)
        # Unpack little‐endian signed chars ('b')
        softmax_output = struct.unpack(f'<{WWD_OUTPUT_DIM}b', bytes(raw))
        # Convert to a NumPy array of dtype int8 (range is -128..127)
        return np.array(softmax_output, dtype=np.int8)

    def get_dequantize_output(self, unprotected_protected_cmd):
        WWD_OUTPUT_DIM = 2
        self.clear_serial()
        self.target.send_cmd(unprotected_protected_cmd, 0x80, bytearray([]))
        # Read back the 2 floats = 8 bytes
        dequantize_output = self.target.simpleserial_read(cmd='e', pay_len=WWD_OUTPUT_DIM*4, timeout=2000)
        # Unpack little-endian floats
        dequantize_output = np.array(struct.unpack('<2f', bytes(dequantize_output)), dtype=np.float32)
        return dequantize_output


    ########################################
    # Tiny CNN
    def send_input_image(self, input_data, unprotected_protected_cmd):
        """Stream one or more images as 62-float (248-byte) packets.

        A single image is flattened to 1-D first so a 3-D array such as (8, 8, 1)
        is not split into 8 tiny packets. 62 floats = 248 bytes stays under the
        252-byte device command buffer (a 64-float / 256-byte packet cannot be
        framed). The preceding send_fault_no_and_cmd adds a settle delay so the
        first packet is not dropped, and the firmware resets its write offset on
        every 0x01 command, so each image lands at the start of the buffer.
        """
        self.clear_serial()
        CHUNK_SIZE = 62

        arr = np.asarray(input_data, dtype=np.float32)
        if arr.size == 64:
            # a single image in any shape: (64,), (8,8), (8,8,1), (1,64), ...
            images = [arr.reshape(-1)]
        elif arr.ndim >= 2:
            # a batch: iterate images, flattening each to 1-D
            images = [a.reshape(-1) for a in arr]
        else:
            images = [arr.reshape(-1)]

        for data in images:
            for offset in range(0, len(data), CHUNK_SIZE):
                sub = data[offset : offset + CHUNK_SIZE]
                float_bytes = struct.pack('<%df' % len(sub), *sub)
                # scmd=0x02 means "image data"
                self.target.send_cmd(unprotected_protected_cmd, 0x02, float_bytes)
                # throttle a bit so the SAM4S can keep up
                time.sleep(0.005)

    def run_image_cnn(self, unprotected_protected_cmd):
        self.clear_serial()
        self.target.send_cmd(unprotected_protected_cmd, 0x04, bytearray([]))
        time.sleep(1)
        return self.target.simpleserial_read(cmd='p', pay_len=1, timeout=0)[0]

    def run_image_cnn_without_return(self, unprotected_protected_cmd):
        self.clear_serial()
        self.target.send_cmd(unprotected_protected_cmd, 0x04, bytearray([]))
        time.sleep(1)
    
    def predict_image_cnn(self, input_data, unprotected_protected_cmd):
        self.send_input_image(input_data, unprotected_protected_cmd)
        return self.run_image_cnn(unprotected_protected_cmd)

    def predict_image_cnn_without_return(self, input_data, unprotected_protected_cmd):
        self.send_input_image(input_data, unprotected_protected_cmd)
        self.run_image_cnn(unprotected_protected_cmd)

    def get_convolution_output(self, unprotected_protected_cmd):
        TOTAL    = 288
        CHUNK    = 144
        n_chunks = TOTAL // CHUNK
    
        raw = bytearray()
        for i in range(n_chunks):
            self.target.send_cmd(unprotected_protected_cmd, 0x08, bytearray([]))
            chunk = self.target.simpleserial_read('a', pay_len=CHUNK, timeout=2000)
            if chunk is None:
                raise IOError(f"Timeout reading chunk {i}: expected {CHUNK} bytes")
            raw.extend(chunk)
    
        if len(raw) != TOTAL:
            raise IOError(f"Expected {TOTAL} bytes, got {len(raw)}")
    
        # unpack as 288 signed 8-bit ints
        vals = struct.unpack(f'<{TOTAL}b', bytes(raw))
        return np.array(vals, dtype=np.int8)

    def get_maxpool_output(self, unprotected_protected_cmd):
        MAXPOOL_DIM = 72
        self.clear_serial()
        self.target.send_cmd(unprotected_protected_cmd, 0x10, bytearray([]))
        # Read back the 72 signed‐bytes = 72 bytes
        raw = self.target.simpleserial_read(cmd='b', pay_len=MAXPOOL_DIM, timeout=2000)
        # Unpack little‐endian signed chars ('b')
        maxpool_output = struct.unpack(f'<{MAXPOOL_DIM}b', bytes(raw))
        # Convert to a NumPy array of dtype int8 (range is -128..127)
        return np.array(maxpool_output, dtype=np.int8)

    def get_fc_output(self, unprotected_protected_cmd):
        FC_OUTPUT_DIM = 10
        self.clear_serial()
        self.target.send_cmd(unprotected_protected_cmd, 0x20, bytearray([]))
        # Read back the 10 signed‐bytes = 10 bytes
        raw = self.target.simpleserial_read(cmd='c', pay_len=FC_OUTPUT_DIM, timeout=2000)
        # Unpack little‐endian signed chars ('b')
        fc_output = struct.unpack(f'<{FC_OUTPUT_DIM}b', bytes(raw))
        # Convert to a NumPy array of dtype int8 (range is -128..127)
        return np.array(fc_output, dtype=np.int8)

    def get_dequantize_output_cnn(self, unprotected_protected_cmd):
        DQ_OUTPUT_DIM = 10
        self.clear_serial()
        self.target.send_cmd(unprotected_protected_cmd, 0x40, bytearray([]))
        # Read back the 10 floats = 40 bytes
        dequantize_output = self.target.simpleserial_read(cmd='d', pay_len=DQ_OUTPUT_DIM*4, timeout=2000)
        # Unpack little-endian floats
        dequantize_output = np.array(struct.unpack('<10f', bytes(dequantize_output)), dtype=np.float32)
        return dequantize_output

