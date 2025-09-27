class SortingAlgorithms:
    def __init__(self):
        self.comparisons = 0
        self.swaps = 0

    def reset_stats(self):
        self.comparisons = 0
        self.swaps = 0

    def bubble_sort(self, arr, visualize=False):
        self.reset_stats()
        arr = arr.copy()
        n = len(arr)
        steps = []
        for i in range(n):
            for j in range(0, n - i - 1):
                self.comparisons += 1
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    self.swaps += 1
                    if visualize:
                        steps.append((arr.copy(), j, j + 1, "swap"))
                elif visualize:
                    steps.append((arr.copy(), j, j + 1, "compare"))
        return arr, steps

    def selection_sort(self, arr, visualize=False):
        self.reset_stats()
        arr = arr.copy()
        n = len(arr)
        steps = []
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                self.comparisons += 1
                if arr[j] < arr[min_idx]:
                    min_idx = j
                if visualize:
                    steps.append((arr.copy(), i, j, "compare"))
            if min_idx != i:
                arr[i], arr[min_idx] = arr[min_idx], arr[i]
                self.swaps += 1
                if visualize:
                    steps.append((arr.copy(), i, min_idx, "swap"))
        return arr, steps

    def insertion_sort(self, arr, visualize=False):
        self.reset_stats()
        arr = arr.copy()
        steps = []
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            while j >= 0 and arr[j] > key:
                self.comparisons += 1
                arr[j + 1] = arr[j]
                self.swaps += 1
                j -= 1
                if visualize:
                    steps.append((arr.copy(), j + 1, j + 2, "swap"))
            arr[j + 1] = key
            if visualize:
                steps.append((arr.copy(), j + 1, i, "insert"))
        return arr, steps

    def quick_sort(self, arr, visualize=False, steps=None, low=0, high=None):
        if steps is None:
            self.reset_stats()
            steps = []
            arr = arr.copy()
        if high is None:
            high = len(arr) - 1
        if low < high:
            pi = self._partition(arr, low, high, visualize, steps)
            self.quick_sort(arr, visualize, steps, low, pi - 1)
            self.quick_sort(arr, visualize, steps, pi + 1, high)
        return arr, steps

    def _partition(self, arr, low, high, visualize, steps):
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            self.comparisons += 1
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
                self.swaps += 1
                if visualize:
                    steps.append((arr.copy(), i, j, "swap"))
            elif visualize:
                steps.append((arr.copy(), j, high, "compare"))
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        self.swaps += 1
        if visualize:
            steps.append((arr.copy(), i + 1, high, "swap"))
        return i + 1
