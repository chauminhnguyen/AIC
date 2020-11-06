from .deep_sort import DeepSort


__all__ = ['DeepSort', 'build_tracker']


def build_tracker(cfg, max_dist, max_nms, max_iou, max_age, use_cuda):
    return DeepSort(model_path=cfg.DEEPSORT.REID_CKPT, 
                max_dist=max_dist, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE, 
                nms_max_overlap=max_nms, max_iou_distance=max_iou, 
                max_age=max_age, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET, use_cuda=use_cuda)
    
