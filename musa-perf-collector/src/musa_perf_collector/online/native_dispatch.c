#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include <stdatomic.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>


typedef struct {
    PyObject_HEAD
    vectorcallfunc vectorcall;
    PyObject *baseline;
    PyObject *selected;
    PyObject *forked;
    PyObject *error_callback;
    PyObject *online_instrumentation;
    PyObject *terminal_reason;
    PyObject *dict;
    PyObject *weakreflist;
    _Atomic uint64_t counter;
    uint64_t every;
    uint64_t start;
    uint64_t disabled_at;
    pid_t owner_pid;
    int disabled;
} NativePeriodicDispatcher;


static uint64_t
candidate_count(uint64_t total, uint64_t start, uint64_t every) {
    if (total <= start) {
        return 0;
    }
    return 1 + (total - 1 - start) / every;
}


static void
record_baseline_error(NativePeriodicDispatcher *self, uint64_t launch_index,
                      const char *reason) {
    PyObject *error_type = NULL;
    PyObject *error_value = NULL;
    PyObject *error_traceback = NULL;
    PyObject *index = NULL;
    PyObject *reason_obj = NULL;
    PyObject *type_name = NULL;
    PyObject *callback_result = NULL;

    if (self->error_callback == NULL || self->error_callback == Py_None) {
        return;
    }
    PyErr_Fetch(&error_type, &error_value, &error_traceback);
    PyErr_NormalizeException(&error_type, &error_value, &error_traceback);
    index = PyLong_FromUnsignedLongLong(launch_index);
    reason_obj = PyUnicode_FromString(reason);
    if (error_type != NULL) {
        type_name = PyObject_GetAttrString(error_type, "__name__");
    }
    if (index != NULL && reason_obj != NULL && type_name != NULL) {
        callback_result = PyObject_CallFunctionObjArgs(
            self->error_callback, index, reason_obj, type_name, NULL);
    }
    Py_XDECREF(callback_result);
    Py_XDECREF(index);
    Py_XDECREF(reason_obj);
    Py_XDECREF(type_name);
    if (PyErr_Occurred()) {
        PyErr_Clear();
    }
    PyErr_Restore(error_type, error_value, error_traceback);
}


static PyObject *
call_baseline(NativePeriodicDispatcher *self, PyObject *const *args,
              size_t nargsf, PyObject *kwnames, uint64_t launch_index,
              const char *reason) {
    PyObject *result = PyObject_Vectorcall(self->baseline, args, nargsf, kwnames);
    if (result == NULL) {
        record_baseline_error(self, launch_index, reason);
    }
    return result;
}


static PyObject *
native_dispatch_vectorcall(PyObject *callable, PyObject *const *args,
                           size_t nargsf, PyObject *kwnames) {
    NativePeriodicDispatcher *self = (NativePeriodicDispatcher *)callable;
    uint64_t launch_index = atomic_fetch_add_explicit(
        &self->counter, 1, memory_order_relaxed);

    if (getpid() != self->owner_pid) {
        return PyObject_Vectorcall(self->forked, args, nargsf, kwnames);
    }
    if (self->disabled) {
        const char *reason = "native_dispatch_terminal_disabled";
        if (self->terminal_reason != NULL && PyUnicode_Check(self->terminal_reason)) {
            const char *value = PyUnicode_AsUTF8(self->terminal_reason);
            if (value != NULL) {
                reason = value;
            } else {
                PyErr_Clear();
            }
        }
        return call_baseline(self, args, nargsf, kwnames, launch_index, reason);
    }
    if (launch_index < self->start ||
        (launch_index - self->start) % self->every != 0) {
        return call_baseline(self, args, nargsf, kwnames, launch_index,
                             "launch_not_selected");
    }

    Py_ssize_t positional = PyVectorcall_NARGS(nargsf);
    Py_ssize_t keyword_count = kwnames == NULL ? 0 : PyTuple_GET_SIZE(kwnames);
    Py_ssize_t total_args = positional + keyword_count;
    PyObject **selected_args = PyMem_Malloc(
        (size_t)(total_args + 1) * sizeof(PyObject *));
    if (selected_args == NULL) {
        return PyErr_NoMemory();
    }
    PyObject *index = PyLong_FromUnsignedLongLong(launch_index);
    if (index == NULL) {
        PyMem_Free(selected_args);
        return NULL;
    }
    selected_args[0] = index;
    if (total_args > 0) {
        memcpy(selected_args + 1, args, (size_t)total_args * sizeof(PyObject *));
    }
    PyObject *selected_result = PyObject_Vectorcall(
        self->selected, selected_args, (size_t)(positional + 1), kwnames);
    Py_DECREF(index);
    PyMem_Free(selected_args);
    if (selected_result == NULL) {
        return NULL;
    }
    if (!PyTuple_Check(selected_result) || PyTuple_GET_SIZE(selected_result) != 2) {
        Py_DECREF(selected_result);
        PyErr_SetString(PyExc_RuntimeError,
                        "native selected callback must return (result, terminal_reason)");
        return NULL;
    }
    PyObject *result = Py_NewRef(PyTuple_GET_ITEM(selected_result, 0));
    PyObject *terminal_reason = PyTuple_GET_ITEM(selected_result, 1);
    if (terminal_reason != Py_None) {
        if (!PyUnicode_Check(terminal_reason)) {
            Py_DECREF(result);
            Py_DECREF(selected_result);
            PyErr_SetString(PyExc_TypeError,
                            "native terminal_reason must be str or None");
            return NULL;
        }
        Py_XSETREF(self->terminal_reason, Py_NewRef(terminal_reason));
        self->disabled_at = launch_index;
        self->disabled = 1;
    }
    Py_DECREF(selected_result);
    return result;
}


static int
native_dispatch_init(NativePeriodicDispatcher *self, PyObject *args,
                     PyObject *kwargs) {
    PyObject *baseline = NULL;
    PyObject *selected = NULL;
    PyObject *forked = NULL;
    PyObject *error_callback = Py_None;
    PyObject *instrumentation = Py_None;
    unsigned long long every = 0;
    unsigned long long start = 0;
    int owner_pid = 0;
    static char *keywords[] = {
        "baseline", "selected", "forked", "every", "start", "owner_pid",
        "error_callback", "online_instrumentation", NULL};
    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "OOOKKi|OO:NativePeriodicDispatcher", keywords,
            &baseline, &selected, &forked, &every, &start, &owner_pid,
            &error_callback, &instrumentation)) {
        return -1;
    }
    if (!PyCallable_Check(baseline) || !PyCallable_Check(selected) ||
        !PyCallable_Check(forked)) {
        PyErr_SetString(PyExc_TypeError,
                        "baseline, selected, and forked must be callable");
        return -1;
    }
    if (error_callback != Py_None && !PyCallable_Check(error_callback)) {
        PyErr_SetString(PyExc_TypeError, "error_callback must be callable or None");
        return -1;
    }
    if (every == 0) {
        PyErr_SetString(PyExc_ValueError, "every must be positive");
        return -1;
    }
    if (owner_pid <= 0) {
        PyErr_SetString(PyExc_ValueError, "owner_pid must be positive");
        return -1;
    }
    self->baseline = Py_NewRef(baseline);
    self->selected = Py_NewRef(selected);
    self->forked = Py_NewRef(forked);
    self->error_callback = Py_NewRef(error_callback);
    self->online_instrumentation = Py_NewRef(instrumentation);
    self->terminal_reason = Py_NewRef(Py_None);
    self->every = (uint64_t)every;
    self->start = (uint64_t)start;
    self->owner_pid = (pid_t)owner_pid;
    self->disabled = 0;
    self->disabled_at = 0;
    atomic_init(&self->counter, 0);
    self->vectorcall = native_dispatch_vectorcall;
    return 0;
}


static int
native_dispatch_traverse(NativePeriodicDispatcher *self, visitproc visit,
                         void *arg) {
    Py_VISIT(self->baseline);
    Py_VISIT(self->selected);
    Py_VISIT(self->forked);
    Py_VISIT(self->error_callback);
    Py_VISIT(self->online_instrumentation);
    Py_VISIT(self->terminal_reason);
    Py_VISIT(self->dict);
    return 0;
}


static int
native_dispatch_clear(NativePeriodicDispatcher *self) {
    Py_CLEAR(self->baseline);
    Py_CLEAR(self->selected);
    Py_CLEAR(self->forked);
    Py_CLEAR(self->error_callback);
    Py_CLEAR(self->online_instrumentation);
    Py_CLEAR(self->terminal_reason);
    Py_CLEAR(self->dict);
    return 0;
}


static void
native_dispatch_dealloc(NativePeriodicDispatcher *self) {
    PyObject_GC_UnTrack(self);
    if (self->weakreflist != NULL) {
        PyObject_ClearWeakRefs((PyObject *)self);
    }
    native_dispatch_clear(self);
    Py_TYPE(self)->tp_free((PyObject *)self);
}


static PyObject *
native_dispatch_summary(NativePeriodicDispatcher *self, PyObject *Py_UNUSED(ignored)) {
    uint64_t total = atomic_load_explicit(&self->counter, memory_order_relaxed);
    uint64_t candidates = candidate_count(total, self->start, self->every);
    uint64_t ordinary_non_candidates = total - candidates;
    uint64_t disabled_candidates = 0;
    if (self->disabled) {
        uint64_t active_total = total < self->disabled_at + 1
                                    ? total : self->disabled_at + 1;
        disabled_candidates = candidates - candidate_count(
            active_total, self->start, self->every);
    }
    uint64_t fast = ordinary_non_candidates + disabled_candidates;
    PyObject *reasons = PyDict_New();
    PyObject *summary = PyDict_New();
    if (reasons == NULL || summary == NULL) {
        Py_XDECREF(reasons);
        Py_XDECREF(summary);
        return NULL;
    }
    if (ordinary_non_candidates > 0) {
        PyObject *count = PyLong_FromUnsignedLongLong(ordinary_non_candidates);
        if (count == NULL || PyDict_SetItemString(reasons, "launch_not_selected", count) < 0) {
            Py_XDECREF(count);
            Py_DECREF(reasons);
            Py_DECREF(summary);
            return NULL;
        }
        Py_DECREF(count);
    }
    if (disabled_candidates > 0) {
        PyObject *count = PyLong_FromUnsignedLongLong(disabled_candidates);
        if (count == NULL || PyDict_SetItem(reasons, self->terminal_reason, count) < 0) {
            Py_XDECREF(count);
            Py_DECREF(reasons);
            Py_DECREF(summary);
            return NULL;
        }
        Py_DECREF(count);
    }
    PyObject *total_obj = PyLong_FromUnsignedLongLong(total);
    PyObject *fast_obj = PyLong_FromUnsignedLongLong(fast);
    if (total_obj == NULL || fast_obj == NULL ||
        PyDict_SetItemString(summary, "total_launch_count", total_obj) < 0 ||
        PyDict_SetItemString(summary, "fast_path_launch_count", fast_obj) < 0 ||
        PyDict_SetItemString(summary, "fast_path_skip_reasons", reasons) < 0) {
        Py_XDECREF(total_obj);
        Py_XDECREF(fast_obj);
        Py_DECREF(reasons);
        Py_DECREF(summary);
        return NULL;
    }
    Py_DECREF(total_obj);
    Py_DECREF(fast_obj);
    Py_DECREF(reasons);
    return summary;
}


static PyMethodDef native_dispatch_methods[] = {
    {"online_launch_summary", (PyCFunction)native_dispatch_summary, METH_NOARGS,
     "Return aggregate launch counts without per-launch Python records."},
    {NULL, NULL, 0, NULL},
};


static PyMemberDef native_dispatch_members[] = {
    {"online_instrumentation", T_OBJECT_EX,
     offsetof(NativePeriodicDispatcher, online_instrumentation), READONLY,
     "Online instrumentation metadata."},
    {NULL, 0, 0, 0, NULL},
};


static PyObject *
native_dispatch_getattro(PyObject *object, PyObject *name) {
    PyObject *result = PyObject_GenericGetAttr(object, name);
    if (result != NULL || !PyErr_ExceptionMatches(PyExc_AttributeError)) {
        return result;
    }
    PyErr_Clear();
    NativePeriodicDispatcher *self = (NativePeriodicDispatcher *)object;
    return PyObject_GetAttr(self->baseline, name);
}


static PyTypeObject NativePeriodicDispatcherType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "musa_perf_collector.online._native_dispatch.NativePeriodicDispatcher",
    .tp_basicsize = sizeof(NativePeriodicDispatcher),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)native_dispatch_dealloc,
    .tp_call = PyVectorcall_Call,
    .tp_getattro = native_dispatch_getattro,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC |
                Py_TPFLAGS_HAVE_VECTORCALL,
    .tp_doc = "Native periodic selector and baseline/instrumented dispatcher.",
    .tp_traverse = (traverseproc)native_dispatch_traverse,
    .tp_clear = (inquiry)native_dispatch_clear,
    .tp_methods = native_dispatch_methods,
    .tp_members = native_dispatch_members,
    .tp_dictoffset = offsetof(NativePeriodicDispatcher, dict),
    .tp_weaklistoffset = offsetof(NativePeriodicDispatcher, weakreflist),
    .tp_vectorcall_offset = offsetof(NativePeriodicDispatcher, vectorcall),
    .tp_init = (initproc)native_dispatch_init,
    .tp_new = PyType_GenericNew,
};


static PyModuleDef native_dispatch_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_native_dispatch",
    .m_doc = "Low-overhead native periodic launch dispatcher.",
    .m_size = -1,
};


PyMODINIT_FUNC
PyInit__native_dispatch(void) {
    if (PyType_Ready(&NativePeriodicDispatcherType) < 0) {
        return NULL;
    }
    PyObject *module = PyModule_Create(&native_dispatch_module);
    if (module == NULL) {
        return NULL;
    }
    Py_INCREF(&NativePeriodicDispatcherType);
    if (PyModule_AddObject(module, "NativePeriodicDispatcher",
                           (PyObject *)&NativePeriodicDispatcherType) < 0) {
        Py_DECREF(&NativePeriodicDispatcherType);
        Py_DECREF(module);
        return NULL;
    }
    return module;
}
