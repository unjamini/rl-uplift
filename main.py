import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


from data_reader import DataReader
from metrics import UMG, SN_UMG_v2
from RLift import RLift


def model_baseline(datas):
    x, y = [], []
    for data in datas:
        if data[1] == 0:
            x.append(data[0])
            y.append(data[2])
    
    baseline_model = RandomForestRegressor(max_depth=5, random_state=0)
    baseline_model.fit(np.array(x), np.array(y))
    return baseline_model


if __name__ == '__main__':
    reader = DataReader(
        path='Kevin_Hillstrom_MineThatData.csv',
        action=['women'],
        label='visit',
    )
    trains, validates, tests = reader.get_datas()
    
    n_feature, n_action = reader.n_feature, reader.n_action
    print(f'Features num: {n_feature}, Actions num: {n_action}')
    
    baseline = model_baseline(trains)
    rlift = RLift(
        name_data=reader.name,
        trains=trains,
        validates=validates,
        tests=tests,
        max_epoch=15,
        hidden_layers=[64],
        n_feature=reader.n_feature,
        size_bag=10000,
        validate_max_steps=1000,
        reward_design='action_depend_baseline',
        n_action=n_action,
        n_bags=10,
        train_eval_func=SN_UMG_v2,
        test_eval_func=UMG,
        model_base=baseline,
        metric='SN_UMG_v2',
        Zero_is_Action=True,
        isLoad=False,
    )
    rlift.train()

    qini_scores = rlift.qini_scores
    epochs = list(range(1, len(qini_scores) + 1))
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, qini_scores, marker='o', linestyle='-', color='b', label='Qini Score')
    plt.title('Qini Score over Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Qini Score', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(epochs)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()