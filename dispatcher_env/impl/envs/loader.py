import numpy as np

class TaskLoader():
    def __init__(self, np_random, number_of_tenants=5):
        self.tenants_number = number_of_tenants
        self.np_random = np_random

        self.future_tasks = None
        self.tasks_data = None

    def reset(self):
        self.future_tasks = self.np_random.integers(10000, 100000, size=self.tenants_number)

    def perform_step(self):
        tasks_per_tenant = np.zeros(self.tenants_number, dtype=int)
        if self.np_random.random() > 0.8:
            picked_tenants = self.np_random.choice(self.tenants_number, size=self.np_random.integers(1, self.tenants_number / 2))
            for tenant in picked_tenants:
                if self.future_tasks[tenant] > 0:
                    tasks_per_tenant[tenant] = self.np_random.integers(1, np.minimum(50, self.future_tasks[tenant]))
                    self.future_tasks[tenant] -= tasks_per_tenant[tenant]
        return tasks_per_tenant

    def is_empty(self):
        return np.sum(self.future_tasks) == 0

    def is_tenant_empty(self, tenant):
        return self.future_tasks[tenant] == 0