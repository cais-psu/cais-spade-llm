#pragma once

// Deterministic ROS field-value encoding; CDR padding is intentionally excluded.
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <rosidl_typesupport_cpp/message_type_support.hpp>
#include <rosidl_typesupport_introspection_cpp/field_types.hpp>
#include <rosidl_typesupport_introspection_cpp/message_introspection.hpp>

namespace preview_fingerprint
{
using Members = rosidl_typesupport_introspection_cpp::MessageMembers;
using Member = rosidl_typesupport_introspection_cpp::MessageMember;

inline void integer(std::vector<unsigned char>& output, std::uint64_t value, std::size_t size)
{
  for (std::size_t i = 0; i < size; ++i)
    output.push_back(static_cast<unsigned char>((value >> (8 * i)) & 0xff));
}

inline void message(std::vector<unsigned char>&, const void*, const Members*);

inline void scalar(std::vector<unsigned char>& output, const void* value, const Member& member)
{
  using namespace rosidl_typesupport_introspection_cpp;
  switch (member.type_id_)
  {
    case ROS_TYPE_BOOLEAN: integer(output, *static_cast<const bool*>(value) ? 1 : 0, 1); break;
    case ROS_TYPE_INT8: integer(output, *static_cast<const std::int8_t*>(value), 1); break;
    case ROS_TYPE_CHAR:
    case ROS_TYPE_OCTET:
    case ROS_TYPE_UINT8: integer(output, *static_cast<const std::uint8_t*>(value), 1); break;
    case ROS_TYPE_INT16: integer(output, *static_cast<const std::int16_t*>(value), 2); break;
    case ROS_TYPE_UINT16: integer(output, *static_cast<const std::uint16_t*>(value), 2); break;
    case ROS_TYPE_INT32: integer(output, *static_cast<const std::int32_t*>(value), 4); break;
    case ROS_TYPE_UINT32: integer(output, *static_cast<const std::uint32_t*>(value), 4); break;
    case ROS_TYPE_INT64: integer(output, *static_cast<const std::int64_t*>(value), 8); break;
    case ROS_TYPE_UINT64: integer(output, *static_cast<const std::uint64_t*>(value), 8); break;
    case ROS_TYPE_FLOAT:
    {
      std::uint32_t bits;
      std::memcpy(&bits, value, sizeof(bits));
      integer(output, bits, sizeof(bits));
      break;
    }
    case ROS_TYPE_DOUBLE:
    {
      std::uint64_t bits;
      std::memcpy(&bits, value, sizeof(bits));
      integer(output, bits, sizeof(bits));
      break;
    }
    case ROS_TYPE_STRING:
    {
      const auto& text = *static_cast<const std::string*>(value);
      integer(output, text.size(), 8);
      output.insert(output.end(), text.begin(), text.end());
      break;
    }
    case ROS_TYPE_MESSAGE:
      message(output, value, static_cast<const Members*>(member.members_->data));
      break;
    default: throw std::invalid_argument("Unsupported ROS field type in scene fingerprint");
  }
}

inline void message(std::vector<unsigned char>& output, const void* object, const Members* members)
{
  for (std::size_t i = 0; i < members->member_count_; ++i)
  {
    const auto& member = members->members_[i];
    const auto* value = static_cast<const unsigned char*>(object) + member.offset_;
    if (!member.is_array_)
      scalar(output, value, member);
    else
    {
      const auto size = member.size_function ? member.size_function(value) : member.array_size_;
      integer(output, size, 8);
      for (std::size_t index = 0; index < size; ++index)
      {
        if (member.type_id_ == rosidl_typesupport_introspection_cpp::ROS_TYPE_BOOLEAN)
        {
          bool item;
          member.fetch_function(value, index, &item);
          scalar(output, &item, member);
        }
        else
          scalar(output, member.get_const_function(value, index), member);
      }
    }
  }
}

template<typename Message>
std::vector<unsigned char> encode(const Message& value)
{
  const auto* support = rosidl_typesupport_cpp::get_message_type_support_handle<Message>();
  const auto* introspection = support->func(support, "rosidl_typesupport_introspection_cpp");
  if (!introspection)
    throw std::invalid_argument("ROS scene introspection is unavailable");
  std::vector<unsigned char> result;
  message(result, &value, static_cast<const Members*>(introspection->data));
  return result;
}
}  // namespace preview_fingerprint
