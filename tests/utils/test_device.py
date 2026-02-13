import math
from unittest.mock import patch

import pytest

from ebm.utils.config import TrainingConfig
from ebm.utils.device import (
    _BASE_BATCH_SIZE,
    _BASE_LR,
    auto_scale_config,
    get_gpu_vram_gb,
    resolve_batch_size,
    scale_lr,
)


class TestGetGpuVramGb:
    def test_returns_none_without_cuda(self):
        with patch('ebm.utils.device.torch.cuda') as mock_cuda:
            mock_cuda.is_available.return_value = False
            assert get_gpu_vram_gb() is None

    def test_returns_vram_with_cuda(self):
        with patch('ebm.utils.device.torch.cuda') as mock_cuda:
            mock_cuda.is_available.return_value = True
            mock_props = type('Props', (), {'total_memory': 80 * 1024**3})()
            mock_cuda.get_device_properties.return_value = mock_props
            result = get_gpu_vram_gb()
            assert result == pytest.approx(80.0, abs=0.1)


class TestResolveBatchSize:
    @pytest.mark.parametrize(
        ('vram_gb', 'expected_batch'),
        [
            (4.0, 256),
            (6.0, 256),
            (12.0, 512),
            (24.0, 1024),
            (32.0, 1024),
            (40.0, 2048),
            (80.0, 4096),
            (141.0, 4096),
        ],
    )
    def test_vram_to_batch_mapping(self, vram_gb: float, expected_batch: int):
        assert resolve_batch_size(vram_gb) == expected_batch


class TestScaleLr:
    def test_same_batch_returns_same_lr(self):
        assert scale_lr(3e-4, 512, 512) == pytest.approx(3e-4)

    def test_double_batch_sqrt2_scaling(self):
        result = scale_lr(3e-4, 512, 1024)
        expected = 3e-4 * math.sqrt(2)
        assert result == pytest.approx(expected)

    def test_quadruple_batch(self):
        result = scale_lr(3e-4, 512, 2048)
        expected = 3e-4 * 2.0
        assert result == pytest.approx(expected)

    def test_half_batch(self):
        result = scale_lr(3e-4, 512, 256)
        expected = 3e-4 * math.sqrt(0.5)
        assert result == pytest.approx(expected)


class TestAutoScaleConfig:
    def test_no_gpu_returns_original(self):
        cfg = TrainingConfig()
        with patch('ebm.utils.device.get_gpu_vram_gb', return_value=None):
            result = auto_scale_config(cfg)
        assert result.batch_size == cfg.batch_size
        assert result.lr == cfg.lr

    def test_batch_size_override(self):
        cfg = TrainingConfig()
        with patch('ebm.utils.device.get_gpu_vram_gb', return_value=None):
            result = auto_scale_config(cfg, batch_size_override=2048)
        assert result.batch_size == 2048
        assert result.lr == pytest.approx(scale_lr(_BASE_LR, _BASE_BATCH_SIZE, 2048))

    def test_gpu_auto_detection(self):
        cfg = TrainingConfig()
        with (
            patch('ebm.utils.device.get_gpu_vram_gb', return_value=141.0),
            patch('ebm.utils.device.torch.cuda.get_device_name', return_value='H200'),
        ):
            result = auto_scale_config(cfg)
        assert result.batch_size == 4096
        assert result.lr == pytest.approx(scale_lr(_BASE_LR, _BASE_BATCH_SIZE, 4096))

    def test_preserves_other_config_fields(self):
        cfg = TrainingConfig(epochs=100, weight_decay=0.05)
        with patch('ebm.utils.device.get_gpu_vram_gb', return_value=None):
            result = auto_scale_config(cfg, batch_size_override=2048)
        assert result.epochs == 100
        assert result.weight_decay == 0.05
