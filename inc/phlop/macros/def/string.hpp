#ifndef _PHLOP_MACROS_DEF_HPP_
#define _PHLOP_MACROS_DEF_HPP_


#define _PHLOP_TO_STR(x) #x
#define PHLOP_TO_STR(x) _PHLOP_TO_STR(x)

#define PHLOP_TO_STR_2(x, y) x##y
#define PHLOP_STR_CAT(x, y) PHLOP_TO_STR_2(x, y)

#endif /*_PHLOP_MACROS_DEF_HPP_*/
