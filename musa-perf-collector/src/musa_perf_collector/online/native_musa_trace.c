#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <musa_runtime_api.h>


enum SlotState {
    SLOT_FREE, SLOT_WRITING, SLOT_DEVICE_READY, SLOT_COPYING,
    SLOT_RESETTING, SLOT_INVALID
};

typedef struct {
    void *device;
    int64_t *host;
    musaEvent_t start;
    musaEvent_t end;
    musaEvent_t copy_done;
    musaEvent_t ready;
    enum SlotState state;
    uint64_t generation;
    int64_t sentinel;
    PyObject *identity;
    int timed;
} NativeTraceSlot;

typedef struct {
    PyObject_HEAD
    NativeTraceSlot *slots;
    int64_t *host_base;
    Py_ssize_t capacity;
    Py_ssize_t extent;
    int device;
    musaStream_t collector_stream;
    uint64_t generation;
    pid_t owner_pid;
    int initialized;
    int stream_created;
    int host_allocated;
    uint64_t event_timing_every;
} NativeMusaTracePool;


static int
raise_musa(const char *operation, musaError_t error) {
    PyErr_Format(PyExc_RuntimeError, "%s failed: %s (%d)", operation,
                 musaGetErrorString(error), (int)error);
    return -1;
}

static void native_trace_release(NativeMusaTracePool *self);


static int
ensure_active(NativeMusaTracePool *self) {
    if (!self->initialized) {
        PyErr_SetString(PyExc_RuntimeError, "native MUSA trace pool is closed");
        return -1;
    }
    return 0;
}


static int
slot_index(NativeMusaTracePool *self, PyObject *value, Py_ssize_t *result) {
    Py_ssize_t index = PyLong_AsSsize_t(value);
    if (index == -1 && PyErr_Occurred()) {
        return -1;
    }
    if (index < 0 || index >= self->capacity) {
        PyErr_SetString(PyExc_IndexError, "native trace slot outside pool");
        return -1;
    }
    *result = index;
    return 0;
}


static void
fill_sentinel(NativeMusaTracePool *self, NativeTraceSlot *slot) {
    for (Py_ssize_t i = 0; i < self->extent; ++i) {
        slot->host[i] = slot->sentinel;
    }
}


static int
native_trace_init(NativeMusaTracePool *self, PyObject *args, PyObject *kwargs) {
    PyObject *pointers = NULL;
    int extent = 0;
    int device = 0;
    unsigned long long timing_every = 1;
    static char *keywords[] = {
        "device_pointers", "trace_extent", "device", "event_timing_every", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oii|K:NativeMusaTracePool",
                                     keywords, &pointers, &extent, &device,
                                     &timing_every)) {
        return -1;
    }
    PyObject *sequence = PySequence_Fast(pointers, "device_pointers must be a sequence");
    if (sequence == NULL) {
        return -1;
    }
    Py_ssize_t capacity = PySequence_Fast_GET_SIZE(sequence);
    if (capacity <= 0 || extent <= 0 || device < 0 || timing_every == 0) {
        Py_DECREF(sequence);
        PyErr_SetString(PyExc_ValueError, "capacity, extent, and device must be valid");
        return -1;
    }
    if ((size_t)capacity > SIZE_MAX / (size_t)extent / sizeof(int64_t)) {
        Py_DECREF(sequence);
        PyErr_SetString(PyExc_OverflowError, "native trace pool size overflows size_t");
        return -1;
    }
    self->slots = PyMem_Calloc((size_t)capacity, sizeof(NativeTraceSlot));
    if (self->slots == NULL) {
        Py_DECREF(sequence);
        return PyErr_NoMemory(), -1;
    }
    self->capacity = capacity;
    self->extent = extent;
    self->device = device;
    self->owner_pid = getpid();
    self->generation = 0;
    self->event_timing_every = (uint64_t)timing_every;
    musaError_t error = musaSetDevice(device);
    if (error != musaSuccess) {
        raise_musa("musaSetDevice", error);
        goto fail;
    }
    error = musaStreamCreateWithFlags(
        &self->collector_stream, musaStreamNonBlocking);
    if (error != musaSuccess) {
        raise_musa("musaStreamCreateWithFlags", error);
        goto fail;
    }
    self->stream_created = 1;
    error = musaHostAlloc(
        (void **)&self->host_base,
        (size_t)capacity * (size_t)extent * sizeof(int64_t),
        musaHostAllocDefault);
    if (error != musaSuccess) {
        raise_musa("musaHostAlloc", error);
        goto fail;
    }
    self->host_allocated = 1;
    for (Py_ssize_t i = 0; i < capacity; ++i) {
        PyObject *item = PySequence_Fast_GET_ITEM(sequence, i);
        unsigned long long pointer = PyLong_AsUnsignedLongLong(item);
        if (pointer == (unsigned long long)-1 && PyErr_Occurred()) {
            goto fail;
        }
        NativeTraceSlot *slot = &self->slots[i];
        slot->device = (void *)(uintptr_t)pointer;
        slot->host = self->host_base + i * extent;
        slot->sentinel = -(INT64_C(1) << 62) + i;
        slot->state = SLOT_FREE;
        error = musaEventCreate(&slot->start);
        if (error != musaSuccess) {
            raise_musa("musaEventCreate(start)", error);
            goto fail;
        }
        error = musaEventCreate(&slot->end);
        if (error != musaSuccess) {
            raise_musa("musaEventCreate(end)", error);
            goto fail;
        }
        error = musaEventCreateWithFlags(
            &slot->copy_done, musaEventDisableTiming);
        if (error != musaSuccess) {
            raise_musa("musaEventCreateWithFlags(copy_done)", error);
            goto fail;
        }
        error = musaEventCreateWithFlags(&slot->ready, musaEventDisableTiming);
        if (error != musaSuccess) {
            raise_musa("musaEventCreateWithFlags(ready)", error);
            goto fail;
        }
        fill_sentinel(self, slot);
        error = musaMemcpyAsync(
            slot->device, slot->host, (size_t)extent * sizeof(int64_t),
            musaMemcpyHostToDevice, self->collector_stream);
        if (error != musaSuccess) {
            raise_musa("musaMemcpyAsync(initial sentinel)", error);
            goto fail;
        }
    }
    error = musaStreamSynchronize(self->collector_stream);
    if (error != musaSuccess) {
        raise_musa("musaStreamSynchronize(initial sentinel)", error);
        goto fail;
    }
    self->initialized = 1;
    Py_DECREF(sequence);
    return 0;

fail:
    Py_DECREF(sequence);
    native_trace_release(self);
    return -1;
}


static PyObject *
native_trace_acquire(NativeMusaTracePool *self, PyObject *identity) {
    if (ensure_active(self) < 0) {
        return NULL;
    }
    if (getpid() != self->owner_pid) {
        Py_RETURN_NONE;
    }
    for (Py_ssize_t i = 0; i < self->capacity; ++i) {
        NativeTraceSlot *slot = &self->slots[i];
        if (slot->state != SLOT_FREE) {
            continue;
        }
        self->generation += 1;
        slot->generation = self->generation;
        slot->timed = ((slot->generation - 1) % self->event_timing_every) == 0;
        slot->state = SLOT_WRITING;
        Py_XSETREF(slot->identity, Py_NewRef(identity));
        return Py_BuildValue("nLK", i, (long long)slot->sentinel,
                             (unsigned long long)slot->generation);
    }
    Py_RETURN_NONE;
}


static PyObject *
native_trace_update_identity(NativeMusaTracePool *self, PyObject *args) {
    if (ensure_active(self) < 0) {
        return NULL;
    }
    PyObject *index_obj = NULL;
    PyObject *identity = NULL;
    if (!PyArg_ParseTuple(args, "OO:update_identity", &index_obj, &identity)) {
        return NULL;
    }
    Py_ssize_t index;
    if (slot_index(self, index_obj, &index) < 0) {
        return NULL;
    }
    if (self->slots[index].state != SLOT_WRITING) {
        PyErr_SetString(PyExc_RuntimeError,
                        "identity update requires a writing slot");
        return NULL;
    }
    Py_XSETREF(self->slots[index].identity, Py_NewRef(identity));
    Py_RETURN_NONE;
}


static PyObject *
native_trace_record_start(NativeMusaTracePool *self, PyObject *args) {
    if (ensure_active(self) < 0) {
        return NULL;
    }
    PyObject *index_obj = NULL;
    unsigned long long stream_value = 0;
    if (!PyArg_ParseTuple(args, "OK:record_start", &index_obj, &stream_value)) {
        return NULL;
    }
    Py_ssize_t index;
    if (slot_index(self, index_obj, &index) < 0) {
        return NULL;
    }
    NativeTraceSlot *slot = &self->slots[index];
    if (slot->state != SLOT_WRITING) {
        PyErr_SetString(PyExc_RuntimeError, "record_start requires a writing slot");
        return NULL;
    }
    if (!slot->timed) {
        Py_RETURN_NONE;
    }
    musaError_t error = musaEventRecord(
        slot->start, (musaStream_t)(uintptr_t)stream_value);
    if (error != musaSuccess) {
        raise_musa("musaEventRecord(start)", error);
        return NULL;
    }
    Py_RETURN_NONE;
}


static PyObject *
native_trace_enqueue(NativeMusaTracePool *self, PyObject *args) {
    if (ensure_active(self) < 0) {
        return NULL;
    }
    PyObject *index_obj = NULL;
    unsigned long long stream_value = 0;
    if (!PyArg_ParseTuple(args, "OK:enqueue_readback", &index_obj, &stream_value)) {
        return NULL;
    }
    Py_ssize_t index;
    if (slot_index(self, index_obj, &index) < 0) {
        return NULL;
    }
    NativeTraceSlot *slot = &self->slots[index];
    if (slot->state != SLOT_WRITING) {
        PyErr_SetString(PyExc_RuntimeError, "enqueue_readback requires a writing slot");
        return NULL;
    }
    musaStream_t producer = (musaStream_t)(uintptr_t)stream_value;
    musaError_t error = musaEventRecord(slot->end, producer);
    if (error != musaSuccess) {
        slot->state = SLOT_INVALID;
        raise_musa("native async trace readback", error);
        return NULL;
    }
    slot->state = SLOT_DEVICE_READY;
    Py_RETURN_NONE;
}


static PyObject *
native_trace_abort(NativeMusaTracePool *self, PyObject *index_obj) {
    if (ensure_active(self) < 0) {
        return NULL;
    }
    Py_ssize_t index;
    if (slot_index(self, index_obj, &index) < 0) {
        return NULL;
    }
    if (self->slots[index].state != SLOT_WRITING &&
        self->slots[index].state != SLOT_INVALID) {
        PyErr_SetString(PyExc_RuntimeError,
                        "abort requires a writing or invalid slot");
        return NULL;
    }
    self->slots[index].state = SLOT_INVALID;
    Py_CLEAR(self->slots[index].identity);
    Py_RETURN_NONE;
}


static PyObject *
native_trace_poll(NativeMusaTracePool *self, PyObject *args, PyObject *kwargs) {
    if (ensure_active(self) < 0) {
        return NULL;
    }
    int wait = 0;
    static char *keywords[] = {"wait", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|p:poll", keywords, &wait)) {
        return NULL;
    }
    PyObject *completed = PyList_New(0);
    if (completed == NULL) {
        return NULL;
    }
    for (Py_ssize_t i = 0; i < self->capacity; ++i) {
        NativeTraceSlot *slot = &self->slots[i];
        if (slot->state == SLOT_RESETTING) {
            musaError_t status = wait ? musaEventSynchronize(slot->ready)
                                      : musaEventQuery(slot->ready);
            if (status == musaSuccess) {
                slot->state = SLOT_FREE;
                Py_CLEAR(slot->identity);
            } else if (status != musaErrorNotReady) {
                Py_DECREF(completed);
                raise_musa("musaEventQuery(ready)", status);
                return NULL;
            }
            continue;
        }
        if (slot->state == SLOT_DEVICE_READY) {
            if (!wait) {
                continue;
            }
            musaError_t status = musaStreamWaitEvent(
                self->collector_stream, slot->end, 0);
            if (status == musaSuccess) status = musaMemcpyAsync(
                slot->host, slot->device, (size_t)self->extent * sizeof(int64_t),
                musaMemcpyDeviceToHost, self->collector_stream);
            if (status == musaSuccess) status = musaEventRecord(
                slot->copy_done, self->collector_stream);
            if (status != musaSuccess) {
                slot->state = SLOT_INVALID;
                Py_DECREF(completed);
                raise_musa("deferred native trace readback", status);
                return NULL;
            }
            slot->state = SLOT_COPYING;
        }
        if (slot->state != SLOT_COPYING) {
            continue;
        }
        musaError_t status = wait ? musaEventSynchronize(slot->copy_done)
                                  : musaEventQuery(slot->copy_done);
        if (status == musaErrorNotReady) {
            continue;
        }
        if (status != musaSuccess) {
            Py_DECREF(completed);
            raise_musa("musaEventQuery(copy_done)", status);
            return NULL;
        }
        PyObject *elapsed_obj = Py_None;
        if (slot->timed) {
            float elapsed = 0.0f;
            status = musaEventElapsedTime(&elapsed, slot->start, slot->end);
            if (status != musaSuccess) {
                Py_DECREF(completed);
                raise_musa("musaEventElapsedTime", status);
                return NULL;
            }
            elapsed_obj = PyFloat_FromDouble((double)elapsed);
            if (elapsed_obj == NULL) {
                Py_DECREF(completed);
                return NULL;
            }
        } else {
            Py_INCREF(elapsed_obj);
        }
        PyObject *values = PyList_New(self->extent);
        if (values == NULL) {
            Py_DECREF(elapsed_obj);
            Py_DECREF(completed);
            return NULL;
        }
        for (Py_ssize_t j = 0; j < self->extent; ++j) {
            PyObject *value = PyLong_FromLongLong(slot->host[j]);
            if (value == NULL) {
                Py_DECREF(values);
                Py_DECREF(elapsed_obj);
                Py_DECREF(completed);
                return NULL;
            }
            PyList_SET_ITEM(values, j, value);
        }
        PyObject *record = Py_BuildValue(
            "{s:n,s:K,s:L,s:O,s:O,s:O}",
            "slot_id", i,
            "generation", (unsigned long long)slot->generation,
            "sentinel", (long long)slot->sentinel,
            "values", values,
            "kernel_event_ms", elapsed_obj,
            "launch_identity", slot->identity == NULL ? Py_None : slot->identity);
        Py_DECREF(values);
        Py_DECREF(elapsed_obj);
        if (record == NULL || PyList_Append(completed, record) < 0) {
            Py_XDECREF(record);
            Py_DECREF(completed);
            return NULL;
        }
        Py_DECREF(record);
        slot->sentinel = -(INT64_C(1) << 62) + (int64_t)self->generation +
                         self->capacity + i;
        fill_sentinel(self, slot);
        status = musaMemcpyAsync(slot->device, slot->host,
                                 (size_t)self->extent * sizeof(int64_t),
                                 musaMemcpyHostToDevice, self->collector_stream);
        if (status == musaSuccess) status = musaEventRecord(slot->ready, self->collector_stream);
        if (status != musaSuccess) {
            slot->state = SLOT_INVALID;
            Py_DECREF(completed);
            raise_musa("native trace reset", status);
            return NULL;
        }
        slot->state = SLOT_RESETTING;
    }
    if (wait) {
        for (Py_ssize_t i = 0; i < self->capacity; ++i) {
            NativeTraceSlot *slot = &self->slots[i];
            if (slot->state == SLOT_RESETTING) {
                musaError_t status = musaEventSynchronize(slot->ready);
                if (status != musaSuccess) {
                    slot->state = SLOT_INVALID;
                    Py_DECREF(completed);
                    raise_musa("musaEventSynchronize(ready)", status);
                    return NULL;
                }
                slot->state = SLOT_FREE;
                Py_CLEAR(slot->identity);
            }
        }
    }
    return completed;
}


static PyObject *
native_trace_pending(NativeMusaTracePool *self, void *Py_UNUSED(closure)) {
    if (ensure_active(self) < 0) {
        return NULL;
    }
    Py_ssize_t count = 0;
    for (Py_ssize_t i = 0; i < self->capacity; ++i) {
        enum SlotState state = self->slots[i].state;
        if (state == SLOT_WRITING || state == SLOT_DEVICE_READY ||
            state == SLOT_COPYING || state == SLOT_RESETTING) {
            count += 1;
        }
    }
    return PyLong_FromSsize_t(count);
}


static void
native_trace_release(NativeMusaTracePool *self) {
    if (getpid() == self->owner_pid) {
        musaSetDevice(self->device);
        if (self->stream_created) {
            musaStreamSynchronize(self->collector_stream);
        }
        for (Py_ssize_t i = 0; i < self->capacity; ++i) {
            NativeTraceSlot *slot = &self->slots[i];
            if (slot->start != NULL) musaEventDestroy(slot->start);
            if (slot->end != NULL) musaEventDestroy(slot->end);
            if (slot->copy_done != NULL) musaEventDestroy(slot->copy_done);
            if (slot->ready != NULL) musaEventDestroy(slot->ready);
            slot->start = NULL;
            slot->end = NULL;
            slot->copy_done = NULL;
            slot->ready = NULL;
        }
        if (self->stream_created) {
            musaStreamDestroy(self->collector_stream);
        }
        if (self->host_allocated) {
            musaFreeHost(self->host_base);
        }
    }
    self->initialized = 0;
    self->stream_created = 0;
    self->host_allocated = 0;
    self->collector_stream = NULL;
    self->host_base = NULL;
}


static PyObject *
native_trace_close(NativeMusaTracePool *self, PyObject *Py_UNUSED(ignored)) {
    native_trace_release(self);
    Py_RETURN_NONE;
}


static void
native_trace_dealloc(NativeMusaTracePool *self) {
    native_trace_release(self);
    for (Py_ssize_t i = 0; i < self->capacity; ++i) {
        Py_XDECREF(self->slots[i].identity);
    }
    PyMem_Free(self->slots);
    Py_TYPE(self)->tp_free((PyObject *)self);
}


static PyMethodDef native_trace_methods[] = {
    {"acquire", (PyCFunction)native_trace_acquire, METH_O, NULL},
    {"update_identity", (PyCFunction)native_trace_update_identity, METH_VARARGS, NULL},
    {"record_start", (PyCFunction)native_trace_record_start, METH_VARARGS, NULL},
    {"enqueue_readback", (PyCFunction)native_trace_enqueue, METH_VARARGS, NULL},
    {"abort", (PyCFunction)native_trace_abort, METH_O, NULL},
    {"poll", (PyCFunction)native_trace_poll, METH_VARARGS | METH_KEYWORDS, NULL},
    {"close", (PyCFunction)native_trace_close, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL},
};

static PyGetSetDef native_trace_getset[] = {
    {"pending_count", (getter)native_trace_pending, NULL, NULL, NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

static PyTypeObject NativeMusaTracePoolType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "musa_perf_collector.online._native_musa_trace.NativeMusaTracePool",
    .tp_basicsize = sizeof(NativeMusaTracePool),
    .tp_dealloc = (destructor)native_trace_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = native_trace_methods,
    .tp_getset = native_trace_getset,
    .tp_init = (initproc)native_trace_init,
    .tp_new = PyType_GenericNew,
};

static PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_native_musa_trace",
    .m_doc = "Native MUSA Event and asynchronous trace-buffer pool.",
    .m_size = -1,
};

PyMODINIT_FUNC
PyInit__native_musa_trace(void) {
    if (PyType_Ready(&NativeMusaTracePoolType) < 0) return NULL;
    PyObject *result = PyModule_Create(&module);
    if (result == NULL) return NULL;
    Py_INCREF(&NativeMusaTracePoolType);
    if (PyModule_AddObject(result, "NativeMusaTracePool",
                           (PyObject *)&NativeMusaTracePoolType) < 0) {
        Py_DECREF(&NativeMusaTracePoolType);
        Py_DECREF(result);
        return NULL;
    }
    return result;
}
