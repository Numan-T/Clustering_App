import heapq


class MaxHeap:

    def __init__(self, heap_list=[]):
        self.heap_list = self.negate_heap_list(heap_list)
        heapq.heapify(self.heap_list)

    def negate_heap_list(self, heap_list):
        if heap_list:
            # If heap consists of tuples
            if type(heap_list[0]) == tuple:
                negated_heap_list = []
                for tuple_i in heap_list:
                    new_element = [-tuple_i[0]] + \
                        [tuple_i[i] for i in range(1, len(tuple_i))]
                    negated_heap_list.append(tuple(new_element))
                return negated_heap_list
            else:
                return [-i for i in heap_list]
        return []

    def heappush(self, item):
        if type(item) == tuple:
            negated_item = [-item[0]] + \
                [item[i] for i in range(1, len(item))]
            negated_item = tuple(negated_item)
        else:
            negated_item = -item
        heapq.heappush(self.heap_list, negated_item)

    def heappop(self):
        smallest_item = heapq.heappop(self.heap_list)
        if type(smallest_item) == tuple:
            largest_item = [-smallest_item[0]] + \
                [smallest_item[i] for i in range(1, len(smallest_item))]
            return tuple(largest_item)
        else:
            largest_item = -smallest_item
            return largest_item

    def get_largest_item(self):
        smallest_item = self.heap_list[0]
        if type(smallest_item) == tuple:
            largest_item = [-smallest_item[0]] + \
                [smallest_item[i] for i in range(1, len(smallest_item))]
            return tuple(largest_item)
        else:
            largest_item = -smallest_item
            return largest_item


"""
# Examples:
my_list = [(5, "Hallo"), (19, "Hallo"), (6, "Hallo"), (22, "Hallo"),
           (0, "Hallo"), (-6, "Hallo"), (2, "Hallo"), (-37, "Hallo")]
max_heap = MaxHeap(my_list)
print(max_heap.heap_list)
max_heap.heappush((69, "Hallo"))

print(max_heap.heappop())
max_heap.heappush((55, "Hallo"))
print(max_heap.heappop())

print(max_heap.heappop())
"""
