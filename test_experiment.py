import unittest
import torch
from experiment import CHANNELS, EVENTS, UNKNOWN, Stateful, generate, window_features


class ExperimentTests(unittest.TestCase):
    def test_oracle_tracks_every_event(self):
        events, labels = generate(8, 50, .4, 123)
        for stream, truth in zip(events, labels):
            state = [UNKNOWN] * CHANNELS
            for event, expected in zip(stream.tolist(), truth.tolist()):
                if event != EVENTS - 1:
                    state[event // 2] = event % 2
                self.assertEqual(state, expected)

    def test_stateful_chunking_matches_full_stream(self):
        torch.manual_seed(1)
        model = Stateful(12).eval()
        events, _ = generate(3, 12, .5, 7)
        with torch.no_grad():
            full, _ = model(events)
            a, state = model(events[:, :5])
            b, _ = model(events[:, 5:], state)
        torch.testing.assert_close(full, torch.cat([a, b], 1))

    def test_window_does_not_see_future(self):
        x = torch.tensor([[0, 1, 2, 3]])
        features = window_features(x, 2)
        changed = window_features(torch.tensor([[0, 1, 7, 8]]), 2)
        torch.testing.assert_close(features[:, :2], changed[:, :2])


if __name__ == '__main__':
    unittest.main()
