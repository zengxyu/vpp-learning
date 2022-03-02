// Generated by Cap'n Proto compiler, DO NOT EDIT
// source: pointcloud_old.capnp

#pragma once

#include <capnp/generated-header-support.h>
#include <kj/windows-sanity.h>

#if CAPNP_VERSION != 7000
#error "Version mismatch between generated code and library headers.  You must use the same version of the Cap'n Proto compiler and library."
#endif


namespace capnp {
namespace schemas {

CAPNP_DECLARE_SCHEMA(86d176f524f8c4a8);

}  // namespace schemas
}  // namespace capnp

namespace vpp_msg {

struct PointcloudOld {
  PointcloudOld() = delete;

  class Reader;
  class Builder;
  class Pipeline;

  struct _capnpPrivate {
    CAPNP_DECLARE_STRUCT_HEADER(86d176f524f8c4a8, 0, 2)
    #if !CAPNP_LITE
    static constexpr ::capnp::_::RawBrandedSchema const* brand() { return &schema->defaultBrand; }
    #endif  // !CAPNP_LITE
  };
};

// =======================================================================================

class PointcloudOld::Reader {
public:
  typedef PointcloudOld Reads;

  Reader() = default;
  inline explicit Reader(::capnp::_::StructReader base): _reader(base) {}

  inline ::capnp::MessageSize totalSize() const {
    return _reader.totalSize().asPublic();
  }

#if !CAPNP_LITE
  inline ::kj::StringTree toString() const {
    return ::capnp::_::structString(_reader, *_capnpPrivate::brand());
  }
#endif  // !CAPNP_LITE

  inline bool hasPoints() const;
  inline  ::capnp::List<double,  ::capnp::Kind::PRIMITIVE>::Reader getPoints() const;

  inline bool hasLabels() const;
  inline  ::capnp::List< ::int64_t,  ::capnp::Kind::PRIMITIVE>::Reader getLabels() const;

private:
  ::capnp::_::StructReader _reader;
  template <typename, ::capnp::Kind>
  friend struct ::capnp::ToDynamic_;
  template <typename, ::capnp::Kind>
  friend struct ::capnp::_::PointerHelpers;
  template <typename, ::capnp::Kind>
  friend struct ::capnp::List;
  friend class ::capnp::MessageBuilder;
  friend class ::capnp::Orphanage;
};

class PointcloudOld::Builder {
public:
  typedef PointcloudOld Builds;

  Builder() = delete;  // Deleted to discourage incorrect usage.
                       // You can explicitly initialize to nullptr instead.
  inline Builder(decltype(nullptr)) {}
  inline explicit Builder(::capnp::_::StructBuilder base): _builder(base) {}
  inline operator Reader() const { return Reader(_builder.asReader()); }
  inline Reader asReader() const { return *this; }

  inline ::capnp::MessageSize totalSize() const { return asReader().totalSize(); }
#if !CAPNP_LITE
  inline ::kj::StringTree toString() const { return asReader().toString(); }
#endif  // !CAPNP_LITE

  inline bool hasPoints();
  inline  ::capnp::List<double,  ::capnp::Kind::PRIMITIVE>::Builder getPoints();
  inline void setPoints( ::capnp::List<double,  ::capnp::Kind::PRIMITIVE>::Reader value);
  inline void setPoints(::kj::ArrayPtr<const double> value);
  inline  ::capnp::List<double,  ::capnp::Kind::PRIMITIVE>::Builder initPoints(unsigned int size);
  inline void adoptPoints(::capnp::Orphan< ::capnp::List<double,  ::capnp::Kind::PRIMITIVE>>&& value);
  inline ::capnp::Orphan< ::capnp::List<double,  ::capnp::Kind::PRIMITIVE>> disownPoints();

  inline bool hasLabels();
  inline  ::capnp::List< ::int64_t,  ::capnp::Kind::PRIMITIVE>::Builder getLabels();
  inline void setLabels( ::capnp::List< ::int64_t,  ::capnp::Kind::PRIMITIVE>::Reader value);
  inline void setLabels(::kj::ArrayPtr<const  ::int64_t> value);
  inline  ::capnp::List< ::int64_t,  ::capnp::Kind::PRIMITIVE>::Builder initLabels(unsigned int size);
  inline void adoptLabels(::capnp::Orphan< ::capnp::List< ::int64_t,  ::capnp::Kind::PRIMITIVE>>&& value);
  inline ::capnp::Orphan< ::capnp::List< ::int64_t,  ::capnp::Kind::PRIMITIVE>> disownLabels();

private:
  ::capnp::_::StructBuilder _builder;
  template <typename, ::capnp::Kind>
  friend struct ::capnp::ToDynamic_;
  friend class ::capnp::Orphanage;
  template <typename, ::capnp::Kind>
  friend struct ::capnp::_::PointerHelpers;
};

#if !CAPNP_LITE
class PointcloudOld::Pipeline {
public:
  typedef PointcloudOld Pipelines;

  inline Pipeline(decltype(nullptr)): _typeless(nullptr) {}
  inline explicit Pipeline(::capnp::AnyPointer::Pipeline&& typeless)
      : _typeless(kj::mv(typeless)) {}

private:
  ::capnp::AnyPointer::Pipeline _typeless;
  friend class ::capnp::PipelineHook;
  template <typename, ::capnp::Kind>
  friend struct ::capnp::ToDynamic_;
};
#endif  // !CAPNP_LITE

// =======================================================================================

inline bool PointcloudOld::Reader::hasPoints() const {
  return !_reader.getPointerField(
      ::capnp::bounded<0>() * ::capnp::POINTERS).isNull();
}
inline bool PointcloudOld::Builder::hasPoints() {
  return !_builder.getPointerField(
      ::capnp::bounded<0>() * ::capnp::POINTERS).isNull();
}
inline  ::capnp::List<double,  ::capnp::Kind::PRIMITIVE>::Reader PointcloudOld::Reader::getPoints() const {
  return ::capnp::_::PointerHelpers< ::capnp::List<double,  ::capnp::Kind::PRIMITIVE>>::get(_reader.getPointerField(
      ::capnp::bounded<0>() * ::capnp::POINTERS));
}
inline  ::capnp::List<double,  ::capnp::Kind::PRIMITIVE>::Builder PointcloudOld::Builder::getPoints() {
  return ::capnp::_::PointerHelpers< ::capnp::List<double,  ::capnp::Kind::PRIMITIVE>>::get(_builder.getPointerField(
      ::capnp::bounded<0>() * ::capnp::POINTERS));
}
inline void PointcloudOld::Builder::setPoints( ::capnp::List<double,  ::capnp::Kind::PRIMITIVE>::Reader value) {
  ::capnp::_::PointerHelpers< ::capnp::List<double,  ::capnp::Kind::PRIMITIVE>>::set(_builder.getPointerField(
      ::capnp::bounded<0>() * ::capnp::POINTERS), value);
}
inline void PointcloudOld::Builder::setPoints(::kj::ArrayPtr<const double> value) {
  ::capnp::_::PointerHelpers< ::capnp::List<double,  ::capnp::Kind::PRIMITIVE>>::set(_builder.getPointerField(
      ::capnp::bounded<0>() * ::capnp::POINTERS), value);
}
inline  ::capnp::List<double,  ::capnp::Kind::PRIMITIVE>::Builder PointcloudOld::Builder::initPoints(unsigned int size) {
  return ::capnp::_::PointerHelpers< ::capnp::List<double,  ::capnp::Kind::PRIMITIVE>>::init(_builder.getPointerField(
      ::capnp::bounded<0>() * ::capnp::POINTERS), size);
}
inline void PointcloudOld::Builder::adoptPoints(
    ::capnp::Orphan< ::capnp::List<double,  ::capnp::Kind::PRIMITIVE>>&& value) {
  ::capnp::_::PointerHelpers< ::capnp::List<double,  ::capnp::Kind::PRIMITIVE>>::adopt(_builder.getPointerField(
      ::capnp::bounded<0>() * ::capnp::POINTERS), kj::mv(value));
}
inline ::capnp::Orphan< ::capnp::List<double,  ::capnp::Kind::PRIMITIVE>> PointcloudOld::Builder::disownPoints() {
  return ::capnp::_::PointerHelpers< ::capnp::List<double,  ::capnp::Kind::PRIMITIVE>>::disown(_builder.getPointerField(
      ::capnp::bounded<0>() * ::capnp::POINTERS));
}

inline bool PointcloudOld::Reader::hasLabels() const {
  return !_reader.getPointerField(
      ::capnp::bounded<1>() * ::capnp::POINTERS).isNull();
}
inline bool PointcloudOld::Builder::hasLabels() {
  return !_builder.getPointerField(
      ::capnp::bounded<1>() * ::capnp::POINTERS).isNull();
}
inline  ::capnp::List< ::int64_t,  ::capnp::Kind::PRIMITIVE>::Reader PointcloudOld::Reader::getLabels() const {
  return ::capnp::_::PointerHelpers< ::capnp::List< ::int64_t,  ::capnp::Kind::PRIMITIVE>>::get(_reader.getPointerField(
      ::capnp::bounded<1>() * ::capnp::POINTERS));
}
inline  ::capnp::List< ::int64_t,  ::capnp::Kind::PRIMITIVE>::Builder PointcloudOld::Builder::getLabels() {
  return ::capnp::_::PointerHelpers< ::capnp::List< ::int64_t,  ::capnp::Kind::PRIMITIVE>>::get(_builder.getPointerField(
      ::capnp::bounded<1>() * ::capnp::POINTERS));
}
inline void PointcloudOld::Builder::setLabels( ::capnp::List< ::int64_t,  ::capnp::Kind::PRIMITIVE>::Reader value) {
  ::capnp::_::PointerHelpers< ::capnp::List< ::int64_t,  ::capnp::Kind::PRIMITIVE>>::set(_builder.getPointerField(
      ::capnp::bounded<1>() * ::capnp::POINTERS), value);
}
inline void PointcloudOld::Builder::setLabels(::kj::ArrayPtr<const  ::int64_t> value) {
  ::capnp::_::PointerHelpers< ::capnp::List< ::int64_t,  ::capnp::Kind::PRIMITIVE>>::set(_builder.getPointerField(
      ::capnp::bounded<1>() * ::capnp::POINTERS), value);
}
inline  ::capnp::List< ::int64_t,  ::capnp::Kind::PRIMITIVE>::Builder PointcloudOld::Builder::initLabels(unsigned int size) {
  return ::capnp::_::PointerHelpers< ::capnp::List< ::int64_t,  ::capnp::Kind::PRIMITIVE>>::init(_builder.getPointerField(
      ::capnp::bounded<1>() * ::capnp::POINTERS), size);
}
inline void PointcloudOld::Builder::adoptLabels(
    ::capnp::Orphan< ::capnp::List< ::int64_t,  ::capnp::Kind::PRIMITIVE>>&& value) {
  ::capnp::_::PointerHelpers< ::capnp::List< ::int64_t,  ::capnp::Kind::PRIMITIVE>>::adopt(_builder.getPointerField(
      ::capnp::bounded<1>() * ::capnp::POINTERS), kj::mv(value));
}
inline ::capnp::Orphan< ::capnp::List< ::int64_t,  ::capnp::Kind::PRIMITIVE>> PointcloudOld::Builder::disownLabels() {
  return ::capnp::_::PointerHelpers< ::capnp::List< ::int64_t,  ::capnp::Kind::PRIMITIVE>>::disown(_builder.getPointerField(
      ::capnp::bounded<1>() * ::capnp::POINTERS));
}

}  // namespace
